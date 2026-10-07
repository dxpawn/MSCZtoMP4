import mido
import pytest

from mscz2mp4.midi import read_midi


def write_midi(path, tracks, tpb=480):
    m = mido.MidiFile(ticks_per_beat=tpb)
    for msgs in tracks:
        tr = mido.MidiTrack()
        tr.extend(msgs)
        m.tracks.append(tr)
    m.save(path)
    return str(path)


def test_two_staves_tempo_and_pedal(tmp_path):
    # 150 bpm set at tick 0 (faster than the 120 bpm default): one beat = 0.4 s
    upper = [mido.MetaMessage('set_tempo', tempo=400000, time=0),
             mido.Message('control_change', control=64, value=127, time=0),
             mido.Message('note_on', note=72, velocity=80, time=0),
             mido.Message('note_off', note=72, velocity=0, time=480),
             mido.Message('control_change', control=64, value=0, time=0)]
    lower = [mido.Message('note_on', note=48, velocity=60, time=480),
             mido.Message('note_on', note=48, velocity=0, time=960)]      # note_on with velocity 0 = off
    notes, pedal, info = read_midi(write_midi(tmp_path / 'a.mid', [upper, lower]))
    assert info == {'staves': 2, 'out_of_range': 0, 'drums': 0}
    assert notes.shape == (2, 5)
    s, e, p, v, staff = notes[0]
    assert (s, e, p, v, staff) == pytest.approx((0.0, 0.4, 72, 80, 0))
    s, e, p, v, staff = notes[1]
    assert (s, e, p, v, staff) == pytest.approx((0.4, 1.2, 48, 60, 1))
    assert pedal == [(0.0, True), (pytest.approx(0.4), False)]


def test_tempo_change_mid_piece(tmp_path):
    tr = [mido.Message('note_on', note=60, velocity=64, time=0),
          mido.MetaMessage('set_tempo', tempo=1000000, time=480),           # after one beat at 120 bpm
          mido.Message('note_off', note=60, time=480)]
    notes, _, _ = read_midi(write_midi(tmp_path / 'b.mid', [tr]))
    assert notes[0, 1] == pytest.approx(0.5 + 1.0)


def test_restrike_closes_held_note(tmp_path):
    tr = [mido.Message('note_on', note=60, velocity=64, time=0),
          mido.Message('note_on', note=60, velocity=70, time=240),
          mido.Message('note_off', note=60, time=240)]
    notes, _, _ = read_midi(write_midi(tmp_path / 'c.mid', [tr]))
    assert notes[:, :2].tolist() == [[0.0, 0.25], [0.25, 0.5]]


def test_skips_drums_and_out_of_range(tmp_path):
    tr = [mido.Message('note_on', note=10, velocity=64, time=0),           # below A0
          mido.Message('note_off', note=10, time=480),
          mido.Message('note_on', note=60, velocity=64, time=0),
          mido.Message('note_off', note=60, time=480)]
    drums = [mido.Message('note_on', channel=9, note=38, velocity=90, time=0),
             mido.Message('note_off', channel=9, note=38, time=120)]
    notes, _, info = read_midi(write_midi(tmp_path / 'd.mid', [tr, drums]))
    assert info == {'staves': 1, 'out_of_range': 1, 'drums': 1}
    assert notes[:, 2].tolist() == [60]


def test_empty(tmp_path):
    notes, pedal, info = read_midi(write_midi(tmp_path / 'e.mid', [[mido.MetaMessage('end_of_track')]]))
    assert notes.shape == (0, 5) and pedal == [] and info['staves'] == 0
