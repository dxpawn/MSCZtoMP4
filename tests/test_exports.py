import os
import zipfile

import pytest

from mscz2mp4 import exports
from mscz2mp4.tools import ToolError

MSCX = '''<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.70">
  <Score>
    {layout}<Division>480</Division>
    <metaTag name="arranger">Someone &amp; Co</metaTag>
    <metaTag name="composer">Fallback Composer</metaTag>
    <metaTag name="workTitle">Fallback Title</metaTag>
    {vbox}
  </Score>
</museScore>
'''
VBOX = '''<VBox>
      <Text><style>title</style><text><font size="22"/>Đêm Đông</text></Text>
      <Text><style>subtitle</style><text>Line one<br/>Line &amp; two</text></Text>
      <Text><style>composer</style><text>Nguyễn Văn Thương</text></Text>
    </VBox>'''


def make_mscz(path, layout='', vbox=VBOX):
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('score.mscx', MSCX.format(layout=layout, vbox=vbox))
        z.writestr('Excerpts/Part/Part.mscx', '<museScore/>')
        z.writestr('META-INF/container.xml', '<container/>')
    return str(path)


def test_force_line_layout_replaces_existing():
    s = exports.force_line_layout('<layoutMode>page</layoutMode>\n<Division>480</Division>')
    assert s == '<layoutMode>line</layoutMode>\n<Division>480</Division>'


def test_force_line_layout_inserts_before_division():
    s = exports.force_line_layout('<Score>\n    <Division>480</Division>')
    assert '<layoutMode>line</layoutMode>\n    <Division>480</Division>' in s


def test_force_line_layout_rejects_non_scores():
    with pytest.raises(ToolError):
        exports.force_line_layout('<html/>')


def test_meta_from_title_frame(tmp_path):
    m = exports.read_meta(make_mscz(tmp_path / 's.mscz'))
    assert m == {'title': 'Đêm Đông', 'subtitle': 'Line one\nLine & two', 'credits': 'Nguyễn Văn Thương'}


def test_meta_falls_back_to_properties(tmp_path):
    m = exports.read_meta(make_mscz(tmp_path / 's.mscz', vbox=''))
    assert m == {'title': 'Fallback Title', 'subtitle': '', 'credits': 'Fallback Composer\nSomeone & Co'}


def test_meta_normalises_decomposed_unicode(tmp_path):
    decomposed = 'Nguyễn'                     # e + circumflex + tilde as combining marks
    vbox = f'<VBox><Text><style>title</style><text>{decomposed}</text></Text></VBox>'
    assert exports.read_meta(make_mscz(tmp_path / 's.mscz', vbox=vbox))['title'] == 'Nguyễn'


def test_positions(tmp_path):
    p = tmp_path / 'score.spos'
    p.write_text('''<score><elements>
        <element id="0" x="10" y="0" sx="5" sy="20" page="0"/>
        <element id="1" x="15" y="0" sx="7" sy="20" page="0"/>
      </elements><events>
        <event elid="1" position="1500"/>
        <event elid="0" position="0"/>
      </events></score>''')
    assert exports.positions(str(p)) == [(0.0, 10.0, 5.0), (1.5, 15.0, 7.0)]


def test_default_cache_dir(tmp_path):
    score = str(tmp_path / 'My Piece.mscz')
    assert exports.default_cache_dir(score) == os.path.join(str(tmp_path), 'mscz2mp4-cache', 'My Piece')
    assert exports.default_cache_dir(score, 'elsewhere') == os.path.join('elsewhere', 'My Piece')


def test_prepare_patches_copy_and_keeps_foreign_files(tmp_path, monkeypatch):
    score = make_mscz(tmp_path / 's.mscz', layout='<layoutMode>page</layoutMode>\n    ')
    build = tmp_path / 'cache'
    build.mkdir()
    (build / 'notes.txt').write_text('mine')
    (build / 'score.mid').write_text('stale')
    calls = []

    def fake_run(cmd, args):
        calls.append(args)
        open(args[-2], 'w').close()                      # '-o', out, line
    monkeypatch.setattr(exports, 'run_musescore', fake_run)

    line = exports.prepare(score, str(build), ['mscore'])
    with zipfile.ZipFile(line) as z:
        assert '<layoutMode>line</layoutMode>' in z.read('score.mscx').decode()
    assert (build / 'notes.txt').read_text() == 'mine'      # never delete files we did not write
    assert len(calls) == 4 and calls[-1][:2] == ['-b', '320']

    calls.clear()
    exports.prepare(score, str(build), ['mscore'])            # unchanged score: everything cached
    assert calls == []


def test_prepare_rejects_non_zip(tmp_path):
    bad = tmp_path / 'x.mscz'
    bad.write_text('not a zip')
    with pytest.raises(ToolError, match='not a .mscz'):
        exports.prepare(str(bad), str(tmp_path / 'cache'), ['mscore'])
