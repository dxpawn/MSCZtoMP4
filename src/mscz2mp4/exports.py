# SPDX-License-Identifier: AGPL-3.0-or-later
"""Everything taken from MuseScore's own exports of the score, so that hidden tempo marks and dynamics play
exactly as they do in MuseScore:

- score strip: a PNG of the score in continuous view (forced on a copy: <layoutMode>line</layoutMode>);
- timing: .spos (x of every segment and its time in ms) and .mpos (bar boxes), the files musescore.com uses
  for score following;
- notes and pedal: .mid, velocities included;
- audio: .mp3 (MuseScore 4's command line fails on .wav/.flac on some systems; mp3 works everywhere).

The exports are cached in one folder per score and redone when the content of the .mscz changes.
"""
import hashlib
import html
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from .tools import ToolError, run_musescore

Image.MAX_IMAGE_PIXELS = None          # the continuous-view strip is one very wide image

CACHE_VERSION = '2'                    # bump when the cached files change meaning
EXPORTS = ('mid', 'mpos', 'spos', 'mp3')
# every file this program writes into a cache folder; nothing else there is ever deleted
OWN_FILE = re.compile(r'(line\.mscz|source\.sha1|score\.(mid|mpos|spos|mp3)|strip\d+(-\d+)?\.png'
                      r'|strip_\d+p\.npz|audio_lag\.txt)$')


def default_cache_dir(score, root=None):
    """<root>/<score name>/, with root defaulting to 'mscz2mp4-cache' next to the score."""
    stem = os.path.splitext(os.path.basename(score))[0]
    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(score)), 'mscz2mp4-cache')
    return os.path.join(root, stem)


def force_line_layout(mscx):
    """Switch a score (.mscx text) to continuous view, so the PNG export is one long single-row strip."""
    if '<layoutMode>' in mscx:
        return re.sub(r'<layoutMode>\w+</layoutMode>', '<layoutMode>line</layoutMode>', mscx, count=1)
    if '<Division>' not in mscx:
        raise ToolError('this does not look like a MuseScore score (no <Division> in the .mscx)')
    return mscx.replace('<Division>', '<layoutMode>line</layoutMode>\n    <Division>', 1)


def main_mscx(zf):
    """Name of the score's own .mscx inside a .mscz (parts live under Excerpts/)."""
    names = [n for n in zf.namelist() if n.endswith('.mscx') and not n.startswith('Excerpts/')]
    if not names:
        raise ToolError('no .mscx found inside the .mscz')
    return names[0]


def prepare(mscz, build, mscore):
    """Copy the score into the cache in continuous view and export everything from that copy (cached).
    Returns the path of the copy."""
    os.makedirs(build, exist_ok=True)
    line = os.path.join(build, 'line.mscz')
    # keyed on content, not dates: a score copied in from elsewhere keeps its old modification time
    with open(mscz, 'rb') as f:
        key = hashlib.sha1(f.read()).hexdigest() + ' v' + CACHE_VERSION
    src = os.path.join(build, 'source.sha1')
    fresh = os.path.exists(line) and os.path.exists(src) and open(src).read() == key
    if not fresh:
        for f in os.listdir(build):
            if OWN_FILE.match(f):
                os.remove(os.path.join(build, f))
        try:
            zi = zipfile.ZipFile(mscz)
        except zipfile.BadZipFile:
            raise ToolError(f'{mscz} is not a .mscz file (MuseScore 4 saves scores as .mscz)')
        with zi, zipfile.ZipFile(line, 'w', zipfile.ZIP_DEFLATED) as zo:
            main = main_mscx(zi)
            for info in zi.infolist():
                data = zi.read(info.filename)
                if info.filename == main:
                    data = force_line_layout(data.decode('utf-8')).encode('utf-8')
                zo.writestr(info, data)
    for ext in EXPORTS:
        out = os.path.join(build, 'score.' + ext)
        if not os.path.exists(out):
            print('exporting', ext, flush=True)
            try:
                run_musescore(mscore, (['-b', '320'] if ext == 'mp3' else []) + ['-o', out, line])
                if not os.path.exists(out):
                    raise ToolError(f'MuseScore reported success but wrote no {ext} file')
            except ToolError as e:
                if ext != 'mp3':
                    raise
                # the picture does not need the audio: carry on, and try again next time
                print(f'warning: MuseScore could not export the audio, so videos will be silent.\n  {e}')
    if not fresh:
        with open(src, 'w') as f:            # written last: an interrupted export is redone next time
            f.write(key)
    return line


def strip_png(line, build, dpi, mscore):
    """PNG of the continuous-view score at MuseScore resolution setting `dpi`."""
    out = os.path.join(build, f'strip{dpi}.png')
    if not os.path.exists(out):
        print('exporting score strip at', dpi, 'dpi', flush=True)
        run_musescore(mscore, ['-r', str(dpi), '-o', out, line])
        os.replace(os.path.join(build, f'strip{dpi}-1.png'), out)     # MuseScore appends the page number
    return out


def load_ink(path):
    """Score PNG (black on white, or on transparent) -> ink coverage 0..1 (float32)."""
    la = np.asarray(Image.open(path).convert('LA'))      # luminance + alpha: the strip can be huge
    ink = la[..., 1].astype(np.float32) * (1 / 255)
    ink *= 1 - la[..., 0].astype(np.float32) * (1 / 255)
    return ink


def measure_scale(ink, bars):
    """Image px per .mpos/.spos position unit of a strip PNG.

    MuseScore's -r option does not give that many px per inch (4.7 renders about r/4), so the scale is measured
    from the final barline, which is the right edge of the rightmost bar. Normally that is the rightmost ink in
    the image, but text can stick out past it (e.g. "D.C. al Fine"), so the barline is found as the place where
    the staff lines end."""
    right = max(x + sx for _, x, sx in bars)
    cols = np.where(ink.max(0) > 0.3)[0]
    k_est = (cols[-1] + 1) / right
    dark = ink > 0.5
    rows = dark.mean(1)
    staff_rows = np.where(rows >= 0.5 * rows.max())[0]     # staff lines run (almost) the whole strip
    if rows.max() < 0.5 * len(cols) / ink.shape[1]:
        return k_est                                         # no staff lines found (e.g. hidden): best effort
    on_staff = np.where(dark[staff_rows].mean(0) >= 0.6)[0]
    k_staff = (on_staff[-1] + 1) / right
    # keep the exact rightmost-ink measurement unless something clearly sticks out past the staff
    return k_est if k_staff >= 0.997 * k_est else k_staff


def positions(path):
    """.spos/.mpos file -> [(time in s, x, width)] sorted by time."""
    r = ET.parse(path).getroot()
    el = {e.get('id'): (float(e.get('x')), float(e.get('sx'))) for e in r.find('elements')}
    ev = sorted((int(e.get('position')) / 1000, el[e.get('elid')]) for e in r.find('events'))
    return [(t, x, sx) for t, (x, sx) in ev]


def _plain(text):
    text = re.sub(r'<br\s*/>', '\n', text)
    return unicodedata.normalize('NFC', html.unescape(re.sub(r'<[^>]+>', '', text))).strip()


def read_meta(mscz):
    """Title, subtitle and credits: from the title frame (VBox) if there is one, else from the score
    properties (workTitle, composer, arranger)."""
    with zipfile.ZipFile(mscz) as z:
        s = z.read(main_mscx(z)).decode('utf-8')
    tags = {k: _plain(v) for k, v in re.findall(r'<metaTag name="(\w+)">([^<]*)</metaTag>', s)}
    texts = {}
    vbox = re.search(r'<VBox>(.*?)</VBox>', s, re.S)
    if vbox:
        for style, body in re.findall(r'<style>(\w+)</style>.*?<text>(.*?)</text>', vbox.group(1), re.S):
            texts.setdefault(style, _plain(body))
    return {'title': texts.get('title') or tags.get('workTitle', ''),
            'subtitle': texts.get('subtitle', ''),
            'credits': texts.get('composer') or
                       '\n'.join(x for x in (tags.get('composer'), tags.get('arranger')) if x)}
