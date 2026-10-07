# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding and running the external programs: MuseScore 4 and ffmpeg."""
import glob
import os
import re
import shutil
import subprocess
import sys


class ToolError(Exception):
    """A problem the user can fix (missing program, bad input); printed without a traceback."""


MSCORE_NAMES = ('mscore4', 'MuseScore4', 'mscore4portable', 'musescore', 'mscore')
UNVERSIONED = ('musescore', 'mscore')      # Linux distributions may still ship MuseScore 3 under these
FLATPAK_ID = 'org.musescore.MuseScore'

MSCORE_HELP = """MuseScore 4 was not found.
Install it from https://musescore.org (it is free), or tell mscz2mp4 where it is:
  mscz2mp4 --mscore "/path/to/MuseScore4" score.mscz
  or set the MUSESCORE environment variable to that path.
The program is MuseScore4.exe on Windows (in the bin folder of the install),
"MuseScore 4.app/Contents/MacOS/mscore" on macOS, and the AppImage or mscore4portable on Linux."""


def _install_locations():
    """Usual places a MuseScore 4 install ends up, per operating system."""
    home = os.path.expanduser('~')
    if sys.platform == 'win32':
        roots = {os.environ.get(v) for v in ('ProgramFiles', 'ProgramW6432', 'ProgramFiles(x86)')}
        roots.add(os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs'))
        return [os.path.join(r, d, 'bin', 'MuseScore4.exe') for r in sorted(x for x in roots if x)
                for d in ('MuseScore 4', 'MuseScore Studio 4', 'MuseScore Studio')]
    if sys.platform == 'darwin':
        return [os.path.join(root, app, 'Contents', 'MacOS', 'mscore')
                for root in ('/Applications', os.path.join(home, 'Applications'))
                for app in ('MuseScore 4.app', 'MuseScore Studio 4.app', 'MuseScore Studio.app')]
    found = []
    for folder in (os.path.join(home, 'Applications'), os.path.join(home, '.local', 'bin'), '/opt'):
        found += sorted(glob.glob(os.path.join(folder, 'MuseScore*.AppImage')), reverse=True)
    return found


def _flatpak():
    if sys.platform.startswith('linux') and shutil.which('flatpak'):
        r = subprocess.run(['flatpak', 'info', FLATPAK_ID], capture_output=True)
        if r.returncode == 0:
            return ['flatpak', 'run', FLATPAK_ID]
    return None


def _explicit(path, what):
    """A path given by the user: an existing file, or a program name on PATH."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    found = shutil.which(path)
    if found:
        return found
    raise ToolError(f'{what}: {path} does not exist')


def find_musescore(explicit=None):
    """MuseScore command (a list, so Flatpak's 'flatpak run …' fits) and where it was found.

    Order: --mscore, the MUSESCORE environment variable, PATH, the usual install locations."""
    if explicit:
        return [_explicit(explicit, '--mscore')], '--mscore'
    env = os.environ.get('MUSESCORE')
    if env:
        return [_explicit(env, 'MUSESCORE environment variable')], 'MUSESCORE environment variable'
    skipped = []
    for name in MSCORE_NAMES:
        found = shutil.which(name)
        if found:
            major = major_version([found]) if name in UNVERSIONED else None
            if major is not None and major < 4:
                skipped.append(f'{found} is MuseScore {major}, which cannot open MuseScore 4 scores')
                continue
            return [found], 'PATH'
    for path in _install_locations():
        if os.path.isfile(path):
            return [path], 'install location'
    flatpak = _flatpak()
    if flatpak:
        return flatpak, 'Flatpak'
    raise ToolError(MSCORE_HELP + ''.join(f'\n(Skipped {s}.)' for s in skipped))


def major_version(cmd):
    """Major version of a MuseScore program, or None if it cannot be told."""
    m = re.search(r'(\d+)\.\d+', version(cmd, '--version'))
    return int(m[1]) if m else None


def musescore_env():
    """MuseScore's command line still starts Qt; without a display (Linux servers, CI) it needs the
    offscreen platform."""
    env = dict(os.environ)
    if sys.platform.startswith('linux') and not (env.get('DISPLAY') or env.get('WAYLAND_DISPLAY')):
        env.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return env


def run_musescore(cmd, args):
    r = subprocess.run(cmd + list(args), capture_output=True, env=musescore_env())
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).decode('utf-8', 'replace').strip().splitlines()[-8:]
        raise ToolError(f'MuseScore failed (exit code {r.returncode}) on: {" ".join(map(str, args))}'
                        + ('\n  ' + '\n  '.join(tail) if tail else ''))


def has_encoders(ffmpeg):
    """Whether this ffmpeg can write the H.264 + AAC used for the video."""
    try:
        out = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True).stdout.decode('utf-8', 'replace')
    except OSError:
        return False
    names = {line.split()[1] for line in out.splitlines() if len(line.split()) > 1}
    return {'libx264', 'aac'} <= names


def _bundled_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # not installed, or no binary for this platform
        return None


def find_ffmpeg(explicit=None):
    """ffmpeg path and where it was found.

    Order: --ffmpeg, the FFMPEG environment variable, PATH (if it can encode H.264 + AAC), the copy that
    ships with the imageio-ffmpeg package."""
    if explicit:
        return _explicit(explicit, '--ffmpeg'), '--ffmpeg'
    env = os.environ.get('FFMPEG')
    if env:
        return _explicit(env, 'FFMPEG environment variable'), 'FFMPEG environment variable'
    on_path = shutil.which('ffmpeg')
    if on_path and has_encoders(on_path):
        return on_path, 'PATH'
    bundled = _bundled_ffmpeg()
    if bundled:
        return bundled, 'imageio-ffmpeg package'
    if on_path:
        return on_path, 'PATH (no libx264/AAC encoder found: encoding may fail)'
    raise ToolError('ffmpeg was not found. Reinstall mscz2mp4 (it depends on imageio-ffmpeg, which ships one), '
                    'install ffmpeg from https://ffmpeg.org, or pass --ffmpeg /path/to/ffmpeg.')


def version(cmd, flag):
    try:
        r = subprocess.run(cmd + [flag], capture_output=True, timeout=60, env=musescore_env())
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'(could not run: {e})'
    lines = (r.stdout or r.stderr).decode('utf-8', 'replace').strip().splitlines()
    return lines[0] if lines else '(no version output)'

