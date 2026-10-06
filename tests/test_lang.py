import ast
import re
from pathlib import Path

import pytest

from labyrint import LabyrinthGame, available_levels, load_level
from labyrint.lang import EN, number, set_language, t

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = ("labyrint/render.py", "play.py", "dashboard.py")


@pytest.fixture
def english():
    set_language("en")
    yield
    set_language("sv")


def translated_literals(path: Path) -> set[str]:
    """Every string literal passed to t() in a source file (both branches of `t(a if x else b)`)."""
    texts = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t" and node.args:
            arg = node.args[0]
            for value in (arg.body, arg.orelse) if isinstance(arg, ast.IfExp) else (arg,):
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    texts.add(value.value)
    return texts


@pytest.mark.parametrize("name", UI_FILES)
def test_every_ui_text_has_an_english_translation(name):
    texts = translated_literals(ROOT / name)
    assert texts, f"no t() calls found in {name}"
    assert sorted(text for text in texts if text not in EN) == []


def test_names_shown_in_the_ui_are_translated():
    names = [load_level(level).name for level in available_levels()] + ["Mus", "Tangentbord"]
    assert [name for name in names if name not in EN] == []


def test_dashboard_texts_are_translated():
    pytest.importorskip("matplotlib")
    from dashboard import PANELS

    texts = [text for title, explanation, series, _ in PANELS
             for text in (title, explanation, *(label for _, label, *_ in series)) if text]
    assert [text for text in texts if text not in EN] == []


def test_translations_keep_the_placeholders():
    for sv, en in EN.items():
        assert sorted(re.findall(r"\{\w+\}", sv)) == sorted(re.findall(r"\{\w+\}", en)), sv


def test_language_switch(english):
    assert t("Tid") == "Time" and t("unknown text") == "unknown text"
    assert number(8967.5) == "8,967.5"
    set_language("sv")
    assert t("Tid") == "Tid" and number(8967.5) == "8 967,5"
    with pytest.raises(ValueError):
        set_language("de")


def test_english_game_window_draws(english):
    pygame = pytest.importorskip("pygame")
    from labyrint.render import Hud, Renderer

    game = LabyrinthGame("pinnar")
    renderer = Renderer(game.level, 1.5)
    renderer.draw(pygame.Surface((renderer.width, renderer.height)), game, Hud(controller="Mus", paused=True))
