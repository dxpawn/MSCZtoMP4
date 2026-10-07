"""End-to-end checks with the real MuseScore and ffmpeg.

They run on examples/demo.mscz and on every .mscz you put in tests/scores/ (that folder is ignored by git, so it
can hold scores you may not redistribute). Without MuseScore (e.g. on CI) they are skipped. MuseScore's exports
are cached in .pytest_cache, so only the first run per score is slow.
"""
import glob
import os
import subprocess

import numpy as np
import pytest

from mscz2mp4 import exports
from mscz2mp4.render import Video
from mscz2mp4.tools import ToolError, find_ffmpeg, find_musescore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES = [os.path.join(ROOT, 'examples', 'demo.mscz')] + sorted(glob.glob(os.path.join(ROOT, 'tests', 'scores', '*.mscz')))

try:
    MSCORE = find_musescore()[0]
except ToolError:
    MSCORE = None

pytestmark = [pytest.mark.musescore,
              pytest.mark.skipif(MSCORE is None, reason='MuseScore 4 not found (set MUSESCORE to run these)')]


@pytest.fixture(scope='module', params=SCORES, ids=lambda p: os.path.basename(p))
def video(request):
    cache = request.config.cache.mkdir('mscz2mp4')
    build = exports.default_cache_dir(request.param, str(cache))
    return Video(request.param, MSCORE, build, 640, 360)


def test_notes_and_positions(video):
    assert len(video.notes) > 0 and video.midi_info['staves'] >= 1
    assert video.end > 0 and len(video.bars) > 0
    assert np.all(np.diff(video.notes[:, 0]) >= 0)
    assert np.isfinite(video.xg).all()


def test_strip_scale_matches_final_barline(video):
    """The right edge of the rightmost bar, scaled, must land where the staff lines end in the strip image."""
    ink = video.ink[:, video.pad:]
    dark = ink > 128
    rows = dark.mean(1)
    staff_rows = np.where(rows >= 0.5 * rows.max())[0]
    staff_end = np.where(dark[staff_rows].mean(0) >= 0.6)[0][-1] + 1
    right = max(x1 for _, _, x1 in video.bars)
    assert abs(staff_end - right) <= 3


def test_title_card(video):
    assert video.meta['title'] and video.title is not None
    assert not video.missing_glyphs, 'characters neither bundled font can draw'


def test_frames(video):
    title = video.frame(1.5)
    playing = video.frame(video.pre + video.end / 2)
    assert title.shape == playing.shape == (360, 640, 3)
    assert playing.mean() > 10 and not np.array_equal(title, playing)


def test_preview_with_audio(video, tmp_path):
    ffmpeg = find_ffmpeg()[0]
    lag = video.audio_lag(ffmpeg)
    assert 0 <= lag <= 0.15
    out = str(tmp_path / 'clip.mp4')
    video.render(out, ffmpeg, fps=30, start=video.pre, dur=1.0, jobs=2)
    info = subprocess.run([ffmpeg, '-hide_banner', '-i', out], capture_output=True).stderr.decode('utf-8', 'replace')
    assert 'Video: h264' in info and 'Audio: aac' in info
    assert not os.path.exists(out + '.part')
