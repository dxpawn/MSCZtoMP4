# SPDX-License-Identifier: AGPL-3.0-or-later
"""The picture: falling notes onto an 88-key keyboard, a scrolling score strip on top, a title card; and the
video: frames rendered in parallel and piped into ffmpeg together with MuseScore's audio."""
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import exports
from .midi import read_midi
from .tools import ToolError

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
DEFAULT_FONTS = tuple(os.path.join(FONT_DIR, f'EBGaramond-{s}.ttf') for s in ('Regular', 'Bold', 'Italic'))
SYMBOL_FONT = os.path.join(FONT_DIR, 'NotoMusic-Regular.ttf')      # fallback for ♯ ♭ ♮ and the like

# ---------------------------------------------------------------- look (BGR)
BG       = (22, 17, 14)
# score strip themes: paper, ink, playhead, current-bar tint (added; negative = darker), edge line under the strip
STRIPS = {
    'dark':  dict(paper=(30, 24, 20), ink=(212, 226, 236), head=(80, 172, 250), tint=9, edge=(52, 44, 38)),
    'light': dict(paper=(228, 240, 246), ink=(24, 26, 30), head=(24, 104, 214), tint=-10, edge=(150, 166, 176)),
}
GUIDE    = (40, 33, 28)         # octave guide lines
RH_COL   = (80, 172, 250)       # amber: top staff
LH_COL   = (214, 178, 88)       # steel blue: every other staff
IVORY    = (222, 232, 238)
EBONY    = (26, 22, 20)
FELT     = (40, 34, 120)        # strip above the keys; brightens with the pedal
TITLE_RGB = (236, 226, 212)


# ---------------------------------------------------------------- keyboard geometry
BLACK_PC = {1: -0.10, 3: 0.10, 6: -0.13, 8: 0.0, 10: 0.13}


def key_layout(W):
    """x0, x1 per MIDI pitch 21..108, and whether it is black."""
    ww = W / 52
    geo, wi = {}, 0
    for p in range(21, 109):
        pc = p % 12
        if pc in BLACK_PC:
            c = wi * ww + BLACK_PC[pc] * ww
            bw = ww * 0.58
            geo[p] = (c - bw / 2, c + bw / 2, True)
        else:
            geo[p] = (wi * ww, (wi + 1) * ww, False)
            wi += 1
    return geo, ww


# ---------------------------------------------------------------- strip motion
def strip_path(seg, bars, px, end, dt, sigma=0.18):
    """Strip x (px) on a time grid: returns (time of the first grid point, x per grid point).

    Follows the .spos segment times, smoothed so the strip glides instead of jumping from segment to segment.
    Repeats and jumps (D.C., D.S., second endings) show up in the .mpos bars, listed in playing order, as a bar
    that does not start where the previous one ended: the strip cuts there instead of gliding across."""
    tol = 0.01 * float(np.median([sx for _, _, sx in bars]))
    cuts, ends = [], []                    # time of each cut, and the x where the strip stops before it
    for (t0, x0, sx0), (t1, x1, _) in zip(bars, bars[1:]):
        if abs(x1 - (x0 + sx0)) > tol:
            cuts.append(t1)
            ends.append(x0 + sx0)
    ends.append(bars[-1][1] + bars[-1][2])  # the last bar played (the Fine bar after a D.C. al Fine)
    starts = [-np.inf] + cuts
    stops = cuts + [end]

    grid = np.arange(-10, end + 20, dt)
    xg = np.empty_like(grid)
    sig = sigma / dt
    kern = np.exp(-0.5 * (np.arange(-4 * sig, 4 * sig + 1) / sig) ** 2)
    kern /= kern.sum()
    n = len(kern) // 2
    for t_start, t_stop, x_end in zip(starts, stops, ends):
        piece = [(t, x) for t, x, _ in seg if t_start <= t < t_stop] + [(t_stop, x_end)]
        ts, xs = zip(*piece)
        # within a piece the strip only moves forward (small backward steps are layout noise)
        ts, xs = np.maximum.accumulate(ts), np.maximum.accumulate(xs) * px
        sel = (grid >= t_start) & (grid < t_stop) if t_stop != end else grid >= t_start
        x = np.interp(grid[sel], ts, xs)
        if len(x):
            xg[sel] = np.convolve(np.pad(x, n, mode='edge'), kern, 'valid')
    return grid[0], xg


# ---------------------------------------------------------------- drawing helpers
SH = 4                    # cv2 sub-pixel shift: coordinates × 16
F = 1 << SH


def rrect(img, x0, y0, x1, y1, r, col):
    """Filled rounded rectangle with sub-pixel edges."""
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    P = lambda v: int(round(v * F))
    cv2.rectangle(img, (P(x0 + r), P(y0)), (P(x1 - r), P(y1)), col, -1, cv2.LINE_AA, SH)
    cv2.rectangle(img, (P(x0), P(y0 + r)), (P(x1), P(y1 - r)), col, -1, cv2.LINE_AA, SH)
    if r >= 0.5:
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (P(cx), P(cy)), P(r), col, -1, cv2.LINE_AA, SH)


def mix(a, b, f):
    return tuple(int(round(x * (1 - f) + y * f)) for x, y in zip(a, b))


def _runs(txt, font_file):
    """Split a line into [text, font file] runs: characters the font lacks (e.g. the sharp in "C♯ minor") are set
    in the bundled music-symbol font."""
    missing = missing_glyphs(txt, font_file)
    runs = []
    for ch in txt:
        f = SYMBOL_FONT if ch in missing else font_file
        if runs and runs[-1][1] == f:
            runs[-1][0] += ch
        else:
            runs.append([ch, f])
    return runs


def text_layer(lines, W, H):
    """[(text, font file, px, (r,g,b))] centred -> BGR and alpha float layers. Lines too wide for the frame
    are set smaller."""
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    laid = []
    for txt, f, px, rgb in lines:
        runs = _runs(txt, f)

        def measure(px):
            fonts = {ff: ImageFont.truetype(ff, px) for ff in {f} | {ff for _, ff in runs}}
            return fonts, sum(d.textlength(t, font=fonts[ff]) for t, ff in runs)
        fonts, w = measure(px)
        if w > 0.92 * W:
            px = max(1, int(px * 0.92 * W / w))
            fonts, w = measure(px)
        laid.append((runs, fonts, fonts[f], px, w, rgb))
    heights = [main.getbbox('ĐÂg')[3] + px * 0.35 for _, _, main, px, _, _ in laid]
    y = (H - sum(heights)) / 2
    for (runs, fonts, main, _, w, rgb), h in zip(laid, heights):
        x, baseline = (W - w) / 2, y + main.getmetrics()[0]
        for t, ff in runs:                       # all runs on the main font's baseline
            d.text((x, baseline), t, font=fonts[ff], fill=rgb + (255,), anchor='ls')
            x += d.textlength(t, font=fonts[ff])
        y += h
    a = np.asarray(img).astype(np.float32) / 255
    return a[..., [2, 1, 0]], a[..., 3:4]


def missing_glyphs(text, font_file):
    """Characters of `text` the font has no glyph for (they would print as empty boxes)."""
    font = ImageFont.truetype(font_file, 32)

    def shape(ch):
        im = Image.new('L', (64, 64))
        ImageDraw.Draw(im).text((8, 8), ch, font=font, fill=255)
        return im.tobytes()

    notdef = shape('\U0010FFFD')             # private-use code point: drawn as the font's "missing" glyph
    return {ch for ch in set(text) if not ch.isspace() and shape(ch) == notdef}


# ---------------------------------------------------------------- renderer
class Video:
    def __init__(self, mscz, mscore, build, W=2560, H=1440, lookahead=2.2, strip='light', fonts=DEFAULT_FONTS,
                 worker=False):
        """Exports (or reuses the cached exports of) the score and prepares everything that does not change
        from frame to frame. `fonts` = (regular, bold, italic) font files. Render workers pass worker=True:
        the cache is then used as it is, even if the score changes on disk during the render."""
        self.args = (mscz, mscore, build, W, H, lookahead, strip, fonts)
        self.W, self.H = W, H
        self.th = STRIPS[strip]
        self.build = build
        line = os.path.join(build, 'line.mscz') if worker else exports.prepare(mscz, build, mscore)
        self.audio = os.path.join(build, 'score.mp3')
        if not os.path.exists(self.audio):
            self.audio = None                   # MuseScore could not export it: silent video
        u = H / 1080

        # layout
        self.band_h = int(0.34 * H)
        self.kb_h = int(0.15 * H)
        self.kb_top = H - self.kb_h
        self.fall_top = self.band_h + int(3 * u)
        self.felt_h = max(2, int(5 * u))
        self.look = lookahead
        self.speed = (self.kb_top - self.felt_h - self.fall_top) / lookahead
        self.play_x = 0.28 * W
        self.u = u

        # notes
        self.notes, self.pedal, self.midi_info = read_midi(os.path.join(build, 'score.mid'))
        if not len(self.notes):
            raise ToolError('the score has no playable notes (MIDI export is empty)')
        self.end = self.notes[:, 1].max()
        v = self.notes[:, 3]
        lo, hi = np.percentile(v, 5), np.percentile(v, 99)
        if hi - lo < 1:                         # one velocity throughout (no dynamics): middle brightness
            self.vel = np.full(len(v), 0.5)
        else:
            self.vel = np.clip((v - lo) / (hi - lo), 0, 1)
        self.max_dur = (self.notes[:, 1] - self.notes[:, 0]).max()
        self.geo, self.ww = key_layout(W)

        # score strip
        seg = exports.positions(os.path.join(build, 'score.spos'))
        bars = exports.positions(os.path.join(build, 'score.mpos'))
        if not seg or not bars:
            raise ToolError('MuseScore exported no note positions for this score')
        target = self.band_h - 2 * 22 * u
        cache = os.path.join(build, f'strip_{H}p.npz')         # scaled strip, so render workers skip the PNGs
        if os.path.exists(cache):
            c = np.load(cache)
            ink, px = c['ink'], float(c['px'])
        else:
            probe = exports.load_ink(exports.strip_png(line, build, 40, mscore))
            rows = np.where(probe.max(1) > 0.25)[0]
            r = 40 * 2 * target / (rows[-1] - rows[0] + 1)      # ~2x supersampling, then scale down
            # very long scores: keep the PNG under ~60 000 px wide
            r = max(10, int(min(r, 40 * 60000 / probe.shape[1], 1200) // 10 * 10))
            ink = exports.load_ink(exports.strip_png(line, build, r, mscore))
            rows = np.where(ink.max(1) > 0.25)[0]
            y0, y1 = max(0, rows[0] - 2), rows[-1] + 3
            k = target / (y1 - y0)
            px = exports.measure_scale(ink, bars) * k            # position units -> strip px
            ink = cv2.resize(ink[y0:y1], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            ink = np.clip(np.rint(ink * 255), 0, 255).astype(np.uint8)   # 8 bits: a quarter of the memory
            np.savez(cache, ink=ink, px=px)
        self.strip_top = int((self.band_h - ink.shape[0]) / 2)
        self.pad = W
        self.ink = np.pad(ink, ((0, 0), (self.pad, self.pad + 2)))

        self.bars = [(t, x * px, (x + sx) * px) for t, x, sx in bars]
        self.dt = 0.005
        self.grid0, self.xg = strip_path(seg, bars, px, self.end, self.dt)

        # static layers
        self.base = np.empty((H, W, 3), np.uint8)
        self.base[:] = BG
        self.base[:self.band_h] = self.th['paper']
        for p in range(21, 109):                               # faint lines at every C and F
            if p % 12 in (0, 5):
                x = int(round(self.geo[p][0]))
                self.base[self.fall_top:self.kb_top, x] = GUIDE
        # soft edges of the score band
        fade = np.ones(W, np.float32)
        e = int(0.06 * W)
        fade[:e] = np.linspace(0, 1, e) ** 1.5
        fade[-e:] = np.linspace(1, 0, e) ** 1.5
        self.fade2d = np.repeat(fade[None, :], self.ink.shape[0], 0)
        self.kb_base = self._keyboard()
        self.black_mask = self._black_mask()

        # title card
        self.meta = m = exports.read_meta(line)
        regular, bold, italic = fonts
        T = [(l, bold, int(76 * u), TITLE_RGB) for l in m['title'].split('\n') if l]
        T += [(l, italic, int(34 * u), TITLE_RGB) for l in m['subtitle'].split('\n') if l]
        credits = [(l, regular, int(30 * u), TITLE_RGB) for l in m['credits'].split('\n') if l]
        if T and credits:
            T.append(('', regular, int(14 * u), TITLE_RGB))
        T += credits
        self.title = text_layer(T, W, self.kb_top - self.fall_top) if T else None
        self.missing_glyphs = set() if worker else set().union(
            *(missing_glyphs(t, f) & missing_glyphs(t, SYMBOL_FONT) for t, f, _, _ in T))

        self.pre = lookahead + 2.0       # seconds of video before the first note sounds
        self.post = 4.5
        self.duration = self.pre + self.end + self.post

    # ---------------- keyboard
    def _keyboard(self):
        kb = np.empty((self.kb_h, self.W, 3), np.uint8)
        kb[:] = (10, 10, 10)
        for p, (x0, x1, black) in self.geo.items():
            if not black:
                rrect(kb, x0 + 0.6, -4, x1 - 0.6, self.kb_h - 2, 3 * self.u, IVORY)
        # slight shading towards the player
        g = np.linspace(0.88, 1.0, self.kb_h, dtype=np.float32)[:, None, None]
        kb = (kb * g).astype(np.uint8)
        for p, (x0, x1, black) in self.geo.items():
            if black:
                rrect(kb, x0, -4, x1, 0.63 * self.kb_h, 2.5 * self.u, EBONY)
                rrect(kb, x0 + 0.18 * (x1 - x0), -4, x1 - 0.18 * (x1 - x0), 0.6 * self.kb_h, 2 * self.u, (52, 46, 44))
        return kb

    def _black_mask(self):
        m = np.zeros((self.kb_h, self.W), np.uint8)
        for p, (x0, x1, black) in self.geo.items():
            if black:
                rrect(m, x0, -4, x1, 0.63 * self.kb_h, 2.5 * self.u, 255)
        return m > 0

    def strip_x(self, t):
        i = (t - self.grid0) / self.dt
        i0 = int(np.clip(np.floor(i), 0, len(self.xg) - 2))
        f = i - i0
        return self.xg[i0] * (1 - f) + self.xg[i0 + 1] * f

    def pedal_down(self, t):
        st = False
        for pt, down in self.pedal:
            if pt > t:
                break
            st = down
        return st

    # ---------------- one frame
    def frame(self, tv):
        """The frame at video time tv (seconds from the start of the video) as a BGR array."""
        t = tv - self.pre                       # music time
        W, H, u = self.W, self.H, self.u
        img = self.base.copy()

        # score band
        sx = self.strip_x(t) - self.play_x + self.pad
        i0 = int(np.floor(sx)); f = sx - i0
        win = self.ink[:, i0:i0 + W + 1].astype(np.float32) * (1 / 255)
        a = cv2.addWeighted(win[:, :W], 1 - f, win[:, 1:], f, 0)
        a = cv2.multiply(a, self.fade2d)
        y0 = self.strip_top
        th = self.th
        band = cv2.merge([cv2.convertScaleAbs(a, alpha=ic - bc, beta=bc) for ic, bc in zip(th['ink'], th['paper'])])
        # current bar: faint tint behind it
        for bt, bx0, bx1 in reversed(self.bars):
            if bt <= t:
                X0 = int(max(0, bx0 - (sx - self.pad))); X1 = int(max(0, min(W, bx1 - (sx - self.pad))))
                if X1 > X0:
                    op, d = (cv2.add, th['tint']) if th['tint'] > 0 else (cv2.subtract, -th['tint'])
                    img[:self.band_h, X0:X1] = op(img[:self.band_h, X0:X1], (d, d, d, 0))
                    band[:, X0:X1] = op(band[:, X0:X1], (d, d, d, 0))
                break
        img[y0:y0 + a.shape[0]] = band
        # playhead
        ph = img[:self.band_h, int(self.play_x) - 6:int(self.play_x) + 7].astype(np.float32)
        prof = np.exp(-0.5 * (np.arange(-6, 7) / (1.6 * u)) ** 2)[None, :, None]
        ph += (np.array(th['head'], np.float32) - ph) * prof * 0.6
        img[:self.band_h, int(self.play_x) - 6:int(self.play_x) + 7] = ph.astype(np.uint8)
        img[self.band_h:self.band_h + max(1, int(u))] = th['edge']

        # falling notes
        N = self.notes
        lo = np.searchsorted(N[:, 0], t - self.max_dur, 'left')
        hi = np.searchsorted(N[:, 0], t + self.look, 'right')
        layer = np.zeros((H - self.fall_top, W, 3), np.uint8)
        top = self.fall_top
        floor = self.kb_top - self.felt_h
        active = []
        for j in range(lo, hi):
            s, e, p, v, staff = N[j]
            if e <= t:
                continue
            p = int(p)
            x0, x1, black = self.geo[p]
            dur = e - s
            e2 = e - min(0.04, 0.12 * dur)
            yb = floor - (s - t) * self.speed
            yt = floor - (e2 - t) * self.speed
            yb = min(yb, floor); yt = max(yt, top - 10)
            col = RH_COL if staff == 0 else LH_COL
            lev = 0.5 + 0.5 * self.vel[j]
            c = tuple(int(x * lev) for x in col)
            inset = 0.07 * (x1 - x0) if not black else 0.0
            if yb - yt >= 1:
                rrect(layer, x0 + inset, yt - top, x1 - inset, yb - top, 4 * u, c)
                hl = mix(c, (255, 255, 255), 0.35)          # light edge on the leading side
                rrect(layer, x0 + inset + 1.2 * u, yb - top - 3 * u, x1 - inset - 1.2 * u, yb - top - 1.2 * u, 1.5 * u, hl)
            if s <= t:
                active.append((p, col, self.vel[j], t - s))
        glow = cv2.resize(layer, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        # flashes where keys are struck
        for p, col, vv, age in active:
            if age < 0.45:
                x0, x1, _ = self.geo[p]
                k = (1 - age / 0.45) ** 2 * (0.6 + 0.6 * vv)
                cx = (x0 + x1) / 2 / 4
                cy = (floor - top) / 4
                cv2.circle(glow, (int(cx * F), int(cy * F)), int(5 * u * F), tuple(int(min(255, x * k)) for x in col),
                           -1, cv2.LINE_AA, SH)
        glow = cv2.GaussianBlur(glow, (0, 0), 3.5 * u)
        glow = cv2.resize(glow, (W, H - top), interpolation=cv2.INTER_LINEAR)
        fall = img[top:]
        fall[:] = cv2.addWeighted(cv2.max(fall, layer), 1.0, glow, 0.9, 0)

        # title card
        fade_in, hold, fade_out = 0.8, self.pre + 1.5, 1.2
        al = min(1, tv / fade_in) * np.clip(1 - (tv - hold) / fade_out, 0, 1)
        if al > 0 and self.title is not None:
            rgb, aa = self.title
            reg = img[top:self.kb_top].astype(np.float32)
            reg += (rgb * 255 - reg) * aa * al
            img[top:self.kb_top] = reg.astype(np.uint8)

        # keyboard
        kb = self.kb_base.copy()
        whites = [(p, c) for p, c, _, _ in active if not self.geo[p][2]]
        blacks = [(p, c) for p, c, _, _ in active if self.geo[p][2]]
        for p, col in whites:
            x0, x1, _ = self.geo[p]
            rrect(kb, x0 + 0.6, -4, x1 - 0.6, self.kb_h - 2, 3 * u, mix(IVORY, col, 0.75))
        if whites:
            kb[self.black_mask] = self.kb_base[self.black_mask]
        for p, col in blacks:
            x0, x1, _ = self.geo[p]
            rrect(kb, x0, -4, x1, 0.63 * self.kb_h, 2.5 * u, mix(EBONY, col, 0.8))
        img[self.kb_top:] = kb
        felt = mix(FELT, (90, 110, 230), 0.55) if self.pedal_down(t) else FELT
        img[self.kb_top - self.felt_h:self.kb_top] = felt

        # fade from / to black
        g = min(1.0, tv / 0.6, (self.duration - tv) / 2.0)
        if g < 1:
            img = (img * max(0.0, g)).astype(np.uint8)
        return img

    def audio_lag(self, ffmpeg):
        """How late the mp3 is against the MIDI, in seconds (MuseScore Basic soundfont: about 50 ms), cached.
        Attack strength of the audio in 1 ms bins (rise of log energy of the differentiated signal) is
        correlated with the MIDI note onsets weighted by velocity; the best-matching shift is the lag."""
        cache = os.path.join(self.build, 'audio_lag.txt')
        if self.audio is None:
            return 0.0
        if os.path.exists(cache):
            return float(open(cache).read())
        sr, hop = 44100, 44
        raw = subprocess.run([ffmpeg, '-v', 'error', '-i', self.audio, '-ac', '1', '-ar', str(sr),
                              '-f', 'f32le', '-'], capture_output=True).stdout
        if len(raw) < sr * 4:
            print('warning: could not decode the audio to measure its delay; assuming none')
            return 0.0
        x = np.diff(np.frombuffer(raw, np.float32))
        x = x[:len(x) // hop * hop].reshape(-1, hop)
        le = np.log((x.astype(np.float64) ** 2).sum(1) + 1e-9)
        flux = np.convolve(np.maximum(0, np.diff(le, prepend=le[0])), np.ones(3) / 3, 'same')
        imp = np.zeros_like(flux)
        idx = np.round(self.notes[:, 0] * sr / hop).astype(int)
        ok = idx < len(imp)
        np.add.at(imp, idx[ok], self.notes[ok, 3])
        if not flux.any():
            return 0.0                          # silent audio: nothing to line up
        lags = range(-100, 300)
        lag = lags[int(np.argmax([np.dot(np.roll(imp, k), flux) for k in lags]))] * hop / sr
        with open(cache, 'w') as f:
            f.write(f'{lag:.4f}')
        print(f'audio is {lag * 1000:+.0f} ms against the MIDI (compensated)')
        return lag

    # ---------------- output
    def render(self, out, ffmpeg, fps=60, start=0.0, dur=None, crf=16, jobs=None):
        """Write the video (or the part from video time `start`, `dur` seconds long) to `out`."""
        t1 = self.duration if dur is None else min(self.duration, start + dur)
        n = int(round((t1 - start) * fps))
        if n <= 0:
            raise ToolError(f'nothing to render: the video ends at music time {self.duration - self.pre:.1f} s')
        a_start = start - self.pre + self.audio_lag(ffmpeg)   # audio time of the first frame
        delay = max(0.0, -a_start)
        af = [f'adelay={int(delay * 1000)}:all=1'] if delay else []
        af.append('apad')                       # audio as long as the video (silence after the music)
        fade = min(2.0, (t1 - start) / 2)
        af.append(f'afade=t=out:st={t1 - start - fade:.3f}:d={fade:.3f}')
        # ffmpeg writes to a temporary name: an interrupted render never leaves a broken file that looks finished
        part = out + '.part'
        cmd = [ffmpeg, '-y', '-v', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{self.W}x{self.H}', '-r', str(fps), '-i', '-']
        if self.audio:
            cmd += ['-ss', f'{max(0.0, a_start):.3f}', '-i', self.audio,
                    '-map', '0:v', '-map', '1:a', '-af', ','.join(af), '-c:a', 'aac', '-b:a', '320k']
        cmd += ['-t', f'{t1 - start:.3f}', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-f', 'mp4', part]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        times = [start + i / fps for i in range(n)]
        workers = jobs or max(1, min(24, (os.cpu_count() or 2) - 2))
        print(f'rendering {n} frames at {self.W}x{self.H}, {fps} fps, on {workers} processes', flush=True)
        t_start = time.monotonic()
        # spawn everywhere: the same behaviour on every OS, and no fork of a process using OpenCV threads
        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                   initializer=_init_worker, initargs=(self.args,))
        done = False
        try:
            # a bounded number of frames in flight: if ffmpeg is slower than the workers, finished frames
            # (11 MB each at 1440p) must not pile up in memory
            nxt = min(n, 4 * workers)
            pending = deque(pool.submit(_worker_frame, tv) for tv in times[:nxt])
            for i in range(n):
                buf = pending.popleft().result()
                if nxt < n:
                    pending.append(pool.submit(_worker_frame, times[nxt]))
                    nxt += 1
                try:
                    ff.stdin.write(buf)
                except OSError:                  # ffmpeg stopped: its exit code and message say why
                    break
                if i % (fps * 10) == 0 and i:
                    el = time.monotonic() - t_start
                    print(f'  {times[i] - self.pre:6.1f} / {t1 - self.pre:.1f} s music time, '
                          f'{100 * i / n:3.0f}%, about {_mmss(el * (n - i) / i)} left', flush=True)
            else:
                done = True
        except BrokenProcessPool:
            raise ToolError('a render process stopped unexpectedly (out of memory?). '
                            'Try fewer processes, e.g. --jobs 4.')
        finally:
            # after an error or Ctrl+C: drop the queued frames; the few being drawn finish quickly
            pool.shutdown(wait=done, cancel_futures=True)
            try:
                ff.stdin.close()
            except OSError:
                pass
            code = ff.wait()
            if not done or code != 0:
                _remove(part)
        if code != 0 or not done:
            code = code - (1 << 32) if code > (1 << 31) else code      # Windows reports it unsigned
            raise ToolError(f'ffmpeg failed (exit code {code}); see its message above')
        try:
            os.replace(part, out)
        except OSError as e:
            _remove(part)
            raise ToolError(f'could not write {out} (is it open in another program?): {e}')
        print(f'wrote {out} in {_mmss(time.monotonic() - t_start)}')


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _mmss(s):
    s = int(round(s))
    return f'{s // 60}:{s % 60:02d}'


# Render workers: each process builds its own Video from the cache (spawn start method on Windows and macOS,
# so these must be importable top-level functions).
_V = None


def _init_worker(args):
    global _V
    cv2.setNumThreads(1)                 # parallelism comes from the processes
    _V = Video(*args, worker=True)


def _worker_frame(tv):
    return _V.frame(tv).tobytes()
