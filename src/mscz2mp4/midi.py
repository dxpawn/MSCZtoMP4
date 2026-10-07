# SPDX-License-Identifier: AGPL-3.0-or-later
"""Notes and sustain pedal from MuseScore's MIDI export (one track per staff, in score order)."""
import mido
import numpy as np

LOWEST, HIGHEST = 21, 108          # A0 .. C8, the 88 keys
DRUMS = 9                          # MIDI channel 10


def read_midi(path):
    """Returns (notes, pedal, info).

    notes: float array of rows (start s, end s, pitch, velocity, staff), sorted by start; staff is the index
    among the tracks that contain notes (0 = top staff).
    pedal: [(time s, down?)] from CC64, sorted.
    info: {'staves': number of staves with notes, 'out_of_range': notes skipped outside the 88 keys,
    'drums': percussion notes skipped}."""
    m = mido.MidiFile(path)
    tempo = [(0, 500000)]
    for tr in m.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == 'set_tempo':
                tempo.append((tick, msg.tempo))
    tempo.sort(key=lambda e: e[0])           # stable: a tempo set at tick 0 replaces the default
    ticks, secs, t0, s0, us = [0], [0.0], 0, 0.0, 500000
    for tk, tp in tempo:
        s0 += (tk - t0) * us / 1e6 / m.ticks_per_beat
        t0, us = tk, tp
        ticks.append(tk); secs.append(s0)
    tempos = [500000] + [tp for _, tp in tempo]

    def sec(tick):
        i = np.searchsorted(ticks, tick, 'right') - 1
        return secs[i] + (tick - ticks[i]) * tempos[i] / 1e6 / m.ticks_per_beat

    notes, pedal = [], []
    info = {'staves': 0, 'out_of_range': 0, 'drums': 0}
    melodic = [tr for tr in m.tracks
               if any(x.type == 'note_on' and x.channel != DRUMS for x in tr)]
    info['staves'] = len(melodic)
    info['drums'] = sum(x.type == 'note_on' and x.velocity > 0 and x.channel == DRUMS
                        for tr in m.tracks for x in tr)
    for staff, tr in enumerate(melodic):
        tick, on = 0, {}
        for msg in tr:
            tick += msg.time
            if msg.type in ('note_on', 'note_off') and msg.channel == DRUMS:
                continue
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note in on:                       # re-strike: close the held one
                    s, v = on.pop(msg.note)
                    notes.append((s, sec(tick), msg.note, v, staff))
                on[msg.note] = (sec(tick), msg.velocity)
            elif msg.type in ('note_off', 'note_on') and msg.note in on:
                s, v = on.pop(msg.note)
                notes.append((s, sec(tick), msg.note, v, staff))
            elif msg.type == 'control_change' and msg.control == 64:
                pedal.append((sec(tick), msg.value >= 64))
    kept = [n for n in notes if LOWEST <= n[2] <= HIGHEST]
    info['out_of_range'] = len(notes) - len(kept)
    kept.sort()
    pedal.sort()
    return np.array(kept, dtype=np.float64).reshape(-1, 5), pedal, info
