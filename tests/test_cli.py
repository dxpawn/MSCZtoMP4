import pytest

from mscz2mp4.cli import main, parse_args


def test_defaults():
    a = parse_args(['x.mscz'])
    assert (a.size, a.fps, a.crf, a.strip) == ((2560, 1440), 60, 16, 'light')


@pytest.mark.parametrize('argv', [[], ['x.mscz', '--res', '1080p'], ['x.mscz', '--res', '1921x1080'],
                                  ['x.mscz', '--font', 'a', 'b', 'c', 'd'], ['x.mscz', '--jobs', '0']])
def test_rejects(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_missing_score_is_a_clear_error(tmp_path, capsys):
    assert main([str(tmp_path / 'nope.mscz')]) == 1
    assert 'does not exist' in capsys.readouterr().err


def test_wrong_extension(tmp_path, capsys):
    f = tmp_path / 'piece.musicxml'
    f.write_text('')
    assert main([str(f)]) == 1
    assert 'save it as .mscz' in capsys.readouterr().err
