import os

import pytest

from mscz2mp4 import tools
from mscz2mp4.tools import ToolError, find_ffmpeg, find_musescore


@pytest.fixture
def clean(monkeypatch):
    """No MuseScore anywhere unless a test puts one there."""
    for var in ('MUSESCORE', 'FFMPEG'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(tools.shutil, 'which', lambda name: None)
    monkeypatch.setattr(tools, '_install_locations', lambda: [])
    monkeypatch.setattr(tools, '_flatpak', lambda: None)


def exe(tmp_path, name):
    p = tmp_path / name
    p.write_text('')
    return str(p)


def test_explicit_beats_everything(clean, tmp_path, monkeypatch):
    a, b = exe(tmp_path, 'a'), exe(tmp_path, 'b')
    monkeypatch.setenv('MUSESCORE', b)
    assert find_musescore(a) == ([a], '--mscore')


def test_environment_variable(clean, tmp_path, monkeypatch):
    b = exe(tmp_path, 'b')
    monkeypatch.setenv('MUSESCORE', b)
    monkeypatch.setattr(tools.shutil, 'which', lambda name: '/usr/bin/' + name)
    assert find_musescore() == ([b], 'MUSESCORE environment variable')


def test_path_in_name_order(clean, monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', lambda name: '/bin/' + name if name in ('musescore', 'mscore4') else None)
    assert find_musescore() == (['/bin/mscore4'], 'PATH')


def test_install_location(clean, tmp_path, monkeypatch):
    missing, present = str(tmp_path / 'nope'), exe(tmp_path, 'MuseScore4.exe')
    monkeypatch.setattr(tools, '_install_locations', lambda: [missing, present])
    assert find_musescore() == ([present], 'install location')


def test_flatpak_last(clean, monkeypatch):
    monkeypatch.setattr(tools, '_flatpak', lambda: ['flatpak', 'run', tools.FLATPAK_ID])
    assert find_musescore()[1] == 'Flatpak'


def test_not_found_explains(clean):
    with pytest.raises(ToolError, match='--mscore'):
        find_musescore()


def test_bad_explicit_path(clean, tmp_path):
    with pytest.raises(ToolError, match='does not exist'):
        find_musescore(str(tmp_path / 'missing'))


def test_install_locations_are_absolute():
    assert all(os.path.isabs(p) for p in tools._install_locations())


def test_ffmpeg_falls_back_to_bundled(clean, monkeypatch):
    monkeypatch.setattr(tools, '_bundled_ffmpeg', lambda: '/pkg/ffmpeg')
    assert find_ffmpeg() == ('/pkg/ffmpeg', 'imageio-ffmpeg package')


def test_ffmpeg_on_path_needs_encoders(clean, monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', lambda name: '/bin/ffmpeg' if name == 'ffmpeg' else None)
    monkeypatch.setattr(tools, '_bundled_ffmpeg', lambda: '/pkg/ffmpeg')
    monkeypatch.setattr(tools, 'has_encoders', lambda f: False)
    assert find_ffmpeg()[0] == '/pkg/ffmpeg'
    monkeypatch.setattr(tools, 'has_encoders', lambda f: True)
    assert find_ffmpeg() == ('/bin/ffmpeg', 'PATH')


def test_offscreen_without_display(monkeypatch):
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    for var in ('DISPLAY', 'WAYLAND_DISPLAY', 'QT_QPA_PLATFORM'):
        monkeypatch.delenv(var, raising=False)
    assert tools.musescore_env()['QT_QPA_PLATFORM'] == 'offscreen'
    monkeypatch.setenv('DISPLAY', ':0')
    assert 'QT_QPA_PLATFORM' not in tools.musescore_env()
