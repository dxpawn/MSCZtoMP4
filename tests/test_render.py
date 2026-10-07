import numpy as np
import pytest

from mscz2mp4.exports import measure_scale
from mscz2mp4.render import DEFAULT_FONTS, SYMBOL_FONT, _runs, missing_glyphs, strip_path, text_layer

# three bars of 200 units from x=100; at 1.5 px per unit the barlines are at px 450, 750 and 1050 (final)
BARS = [(0.0, 100.0, 200.0), (1.0, 300.0, 200.0), (2.0, 500.0, 200.0)]


def strip(extra_ink_past_end=False):
    ink = np.zeros((100, 1300), np.float32)
    ink[20:80:10, 140:1050] = 1                     # staff lines
    for c in (449, 749, 1046, 1047, 1048, 1049):   # barlines, the last one heavy
        ink[20:71, c] = 1
    rng = np.random.default_rng(1)
    for x in rng.integers(160, 1030, 25):          # noteheads with stems
        ink[45:51, x:x + 8] = 1
        ink[20:50, x + 7] = 1
    if extra_ink_past_end:                         # e.g. "D.C. al Fine" sticking out to the right
        ink[0:15, 1020:1200] = 1
    return ink


def test_scale_from_final_barline():
    assert measure_scale(strip(), BARS) == pytest.approx(1.5)


def test_scale_ignores_text_past_final_barline():
    assert measure_scale(strip(extra_ink_past_end=True), BARS) == pytest.approx(1.5)


def seg_for(bars):
    return [(t + d, x + off, 1.0) for t, x, _ in bars for d, off in ((0.0, 10), (1.0, 60))]


def test_strip_moves_forward_and_ends_at_last_barline():
    bars = [(0.0, 0.0, 100.0), (2.0, 100.0, 100.0), (4.0, 200.0, 100.0)]
    t0, xg = strip_path(seg_for(bars), bars, 1.0, 6.0, 0.005)
    at = lambda t: xg[int(round((t - t0) / 0.005))]
    assert np.all(np.diff(xg) >= -1e-9)
    assert at(2.0) == pytest.approx(110, abs=3)
    assert at(25.0) == pytest.approx(300)


def test_strip_cuts_back_at_a_repeat():
    # bars 1-2 played twice, then bar 3
    bars = [(0.0, 0.0, 100.0), (2.0, 100.0, 100.0), (4.0, 0.0, 100.0), (6.0, 100.0, 100.0), (8.0, 200.0, 100.0)]
    t0, xg = strip_path(seg_for(bars), bars, 1.0, 10.0, 0.005)
    at = lambda t: xg[int(round((t - t0) / 0.005))]
    assert at(3.99) == pytest.approx(200, abs=5)    # reaches the repeat barline ...
    assert at(4.0) == pytest.approx(10, abs=5)      # ... and starts over, without gliding back
    assert at(5.0) == pytest.approx(60, abs=5)
    assert at(9.0) == pytest.approx(260, abs=5)


def test_strip_scale_applies():
    bars = [(0.0, 0.0, 100.0), (2.0, 100.0, 100.0)]
    _, a = strip_path(seg_for(bars), bars, 1.0, 4.0, 0.005)
    _, b = strip_path(seg_for(bars), bars, 2.5, 4.0, 0.005)
    assert np.allclose(b, 2.5 * a)


def test_missing_glyphs():
    regular = DEFAULT_FONTS[0]
    assert missing_glyphs('Đêm Đông - Nguyễn', regular) == set()
    assert missing_glyphs('Nocturne 夜想曲', regular) == {'夜', '想', '曲'}


def test_long_title_is_shrunk_to_fit():
    W = 800
    _, alpha = text_layer([('A very long title ' * 6, DEFAULT_FONTS[1], 60, (255, 255, 255))], W, 300)
    cols = np.where(alpha[..., 0].max(0) > 0)[0]
    assert cols[0] > 0 and cols[-1] < W - 1


def test_symbols_fall_back_to_the_music_font():
    regular = DEFAULT_FONTS[0]
    assert _runs('C♯ minor', regular) == [['C', regular], ['♯', SYMBOL_FONT], [' minor', regular]]
    assert _runs('B♭', regular) == [['B', regular], ['♭', SYMBOL_FONT]]
    assert _runs('plain', regular) == [['plain', regular]]
