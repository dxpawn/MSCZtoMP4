# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command line: mscz2mp4 score.mscz [options]."""
import argparse
import os
import platform
import re
import sys

from . import __version__
from .tools import ToolError, find_ffmpeg, find_musescore, has_encoders, version

EXAMPLES = """examples:
  mscz2mp4 "My Piece.mscz"                    full video -> "My Piece.mp4" next to the score
  mscz2mp4 "My Piece.mscz" --still 0 30 95    frames at 0:00, 0:30 and 1:35 of the music -> PNG
  mscz2mp4 "My Piece.mscz" --preview 90 20    20 s clip with audio, from 1:30 of the music
  mscz2mp4 --check                            show which MuseScore, ffmpeg and font will be used
"""


def parse_args(argv=None):
    from .render import STRIPS
    ap = argparse.ArgumentParser(
        prog='mscz2mp4', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
        description='Turn a MuseScore score (.mscz) into a piano video: falling notes onto an 88-key keyboard, '
                    'the score scrolling on top, a title card, and MuseScore\'s own audio.')
    ap.add_argument('score', nargs='?', help='the .mscz file')
    ap.add_argument('-o', '--out', help='output file (default: the score\'s name with .mp4, next to it)')
    ap.add_argument('--res', default='2560x1440',
                    help='WIDTHxHEIGHT (default 2560x1440: YouTube gives 1440p uploads a much higher bitrate '
                         'than 1080p)')
    ap.add_argument('--fps', type=int, default=60, help='frames per second (default 60)')
    ap.add_argument('--crf', type=int, default=16, help='H.264 quality, lower is better (default 16)')
    ap.add_argument('--strip', choices=sorted(STRIPS), default='light',
                    help='score strip colours: light paper (default) or dark')
    only = ap.add_mutually_exclusive_group()
    only.add_argument('--still', type=float, nargs='+', metavar='T',
                      help='only write PNG frames at these music times (seconds)')
    only.add_argument('--preview', type=float, nargs=2, metavar=('START', 'DUR'),
                      help='only write a short clip with audio: START and DUR in seconds of music time')
    ap.add_argument('--mscore', metavar='PATH', help='MuseScore 4 program (default: found automatically)')
    ap.add_argument('--ffmpeg', metavar='PATH', help='ffmpeg program (default: found automatically)')
    ap.add_argument('--cache-dir', metavar='DIR',
                    help='where MuseScore\'s exports are kept (default: mscz2mp4-cache next to the score)')
    ap.add_argument('--font', nargs='+', metavar='TTF',
                    help='title font files: REGULAR [BOLD [ITALIC]] (default: bundled EB Garamond)')
    ap.add_argument('--jobs', type=int, metavar='N', help='render processes (default: CPU count minus 2)')
    ap.add_argument('--check', action='store_true', help='report the programs and font found, then exit')
    ap.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    a = ap.parse_args(argv)
    if not a.check and not a.score:
        ap.error('give a .mscz file (or --check)')
    m = re.fullmatch(r'(\d+)x(\d+)', a.res or '')
    if not m or int(m[1]) < 64 or int(m[2]) < 64 or int(m[1]) % 2 or int(m[2]) % 2:
        ap.error('--res must be WIDTHxHEIGHT with even numbers, e.g. 2560x1440 or 1920x1080')
    a.size = int(m[1]), int(m[2])
    if a.font and len(a.font) > 3:
        ap.error('--font takes at most three files: REGULAR [BOLD [ITALIC]]')
    if a.jobs is not None and a.jobs < 1:
        ap.error('--jobs must be at least 1')
    if a.fps < 1:
        ap.error('--fps must be at least 1')
    if not 0 <= a.crf <= 51:
        ap.error('--crf must be between 0 and 51')
    if a.preview and a.preview[1] <= 0:
        ap.error('--preview DUR must be more than 0')
    return a


def resolve_fonts(paths):
    from .render import DEFAULT_FONTS
    if not paths:
        return DEFAULT_FONTS
    for p in paths:
        if not os.path.isfile(p):
            raise ToolError(f'--font: {p} does not exist')
    regular = paths[0]
    bold = paths[1] if len(paths) > 1 else regular
    italic = paths[2] if len(paths) > 2 else regular
    return regular, bold, italic


def check(a):
    """--check: report what would be used. Exit code 1 if something required is missing."""
    ok = True
    print(f'mscz2mp4 {__version__}, Python {platform.python_version()} on {platform.platform()}')
    try:
        cmd, where = find_musescore(a.mscore)
        print(f'MuseScore: {" ".join(cmd)}\n           found via {where}; {version(cmd, "--version")}')
    except ToolError as e:
        ok = False
        print('MuseScore: NOT FOUND\n  ' + str(e).replace('\n', '\n  '))
    try:
        ff, where = find_ffmpeg(a.ffmpeg)
        enc = 'H.264 + AAC encoders present' if has_encoders(ff) else 'MISSING libx264 or AAC encoder'
        ok &= 'MISSING' not in enc
        print(f'ffmpeg:    {ff}\n           found via {where}; {version([ff], "-version")}; {enc}')
    except ToolError as e:
        ok = False
        print(f'ffmpeg:    NOT FOUND\n  {e}')
    try:
        fonts = resolve_fonts(a.font)
        for role, f in zip(('regular', 'bold', 'italic'), fonts):
            print(f'font:      {role:8s}{f}')
    except ToolError as e:
        ok = False
        print(f'font:      {e}')
    print('all set' if ok else 'something is missing (see above)')
    return 0 if ok else 1


def run(a):
    from . import exports
    from .render import Video
    import cv2

    score = a.score
    if not os.path.isfile(score):
        raise ToolError(f'{score} does not exist')
    if not score.lower().endswith('.mscz'):
        raise ToolError(f'{score} is not a .mscz file. Open it in MuseScore 4 and save it as .mscz first.')
    fonts = resolve_fonts(a.font)
    mscore, _ = find_musescore(a.mscore)
    ffmpeg = None if a.still else find_ffmpeg(a.ffmpeg)[0]
    build = exports.default_cache_dir(score, a.cache_dir)
    try:
        os.makedirs(build, exist_ok=True)
    except OSError as e:
        raise ToolError(f'cannot create the cache folder {build} ({e.strerror}); pick another place with --cache-dir')
    print(f'cache: {build}', flush=True)

    W, H = a.size
    v = Video(score, mscore, build, W, H, strip=a.strip, fonts=fonts)
    info = v.midi_info
    if info['staves'] != 2:
        print(f'note: the score has {info["staves"]} staves with notes; this tool is designed for a two-staff '
              'piano score. The top staff is drawn in amber, every other staff in blue.')
    if info['out_of_range']:
        print(f'note: {info["out_of_range"]} notes outside the piano range (A0-C8) are not drawn.')
    if info['drums']:
        print(f'note: {info["drums"]} percussion notes are not drawn.')
    if v.missing_glyphs:
        chars = ''.join(sorted(v.missing_glyphs))
        print(f'note: the title font cannot draw these characters of the title card: {chars}\n'
              '      pass a font that has them with --font, e.g. Noto Serif CJK for Chinese, Japanese or Korean.')

    stem = os.path.splitext(os.path.basename(score))[0]
    if a.strip != 'light':
        stem += f' ({a.strip})'
    if a.out and os.path.isdir(a.out):                  # -o some/folder: default name inside it
        base = os.path.join(a.out, stem)
    elif a.out:
        base = os.path.splitext(a.out)[0]
    else:
        base = os.path.join(os.path.dirname(score), stem)
    folder = os.path.dirname(os.path.abspath(base))
    if not os.path.isdir(folder):
        raise ToolError(f'the output folder {folder} does not exist')
    if a.still:
        for s in a.still:
            p = f'{base} still {s:g}.png'
            # imencode + tofile: cv2.imwrite fails on non-ASCII paths on Windows
            cv2.imencode('.png', v.frame(s + v.pre))[1].tofile(p)
            print('wrote', p)
    elif a.preview:
        start = max(0.0, a.preview[0] + v.pre)        # the video starts v.pre seconds before the music
        v.render(f'{base} preview {a.preview[0]:g}.mp4', ffmpeg, a.fps, start, a.preview[1], a.crf, a.jobs)
    else:
        print(f'length {v.duration:.0f} s; rendering takes a while (roughly twice the length at 1440p60 on a '
              'desktop CPU)')
        v.render(base + '.mp4', ffmpeg, a.fps, crf=a.crf, jobs=a.jobs)
    return 0


def main(argv=None):
    # score names may contain any characters; a Windows console may not be UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    a = parse_args(argv)
    try:
        return check(a) if a.check else run(a)
    except ToolError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except OSError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('interrupted', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
