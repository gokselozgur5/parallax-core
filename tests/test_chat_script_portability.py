from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_chat_script_is_ascii_english_source():
    source = (SCRIPTS / "chat_qwen.py").read_text(encoding="utf-8")
    assert source.isascii()


def test_chat_script_takes_the_model_path_from_the_command_line():
    source = (SCRIPTS / "chat_qwen.py").read_text(encoding="utf-8")
    assert 'model_path = "/' not in source
    assert any(name in source for name in ("sys.argv", "argparse", "os.environ"))
