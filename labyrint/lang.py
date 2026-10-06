"""UI text in Swedish (the default) or English.

The Swedish text is the key, so the code still reads as Swedish UI text:

    from labyrint.lang import set_language, t
    set_language("en")
    t("Tid")                                  # -> "Time"
    t("Kulan föll i hål {n}!").format(n=3)    # translate a template first, then fill it in

Texts without an English entry are shown unchanged.
"""
from __future__ import annotations

LANGUAGES = ("sv", "en")

EN = {
    # game window (render.py)
    "LABYRINT": "LABYRINTH",
    "Bana: {name}": "Level: {name}",
    "Tid": "Time",
    "Hål passerade": "Holes passed",
    "Försök": "Attempts",
    "Bästa tid": "Best time",
    "Rekord": "Record",
    "{n} hål": "{n} holes",
    "X-ratt": "X knob",
    "Y-ratt": "Y knob",
    "Styrs av: {controller}": "Controlled by: {controller}",
    "PAUS – tryck P": "PAUSED – press P",
    "Mus: luta brädet": "Mouse: tilt the board",
    "Pilar/WASD: vrid rattarna": "Arrows/WASD: turn the knobs",
    "Mellanslag: plant bräde": "Space: level the board",
    "R: börja om    P: paus": "R: restart    P: pause",
    "F1: visa AI-sensorer": "F1: show AI sensors",
    "F2: växla autopilot / AI": "F2: switch autopilot / AI",
    "Esc: avsluta": "Esc: quit",
    # controllers and level names
    "Mus": "Mouse",
    "Tangentbord": "Keyboard",
    "Klassisk": "Classic",
    "Med pinnar": "With pins",
    "Namnlös": "Untitled",
    # play.py
    "Labyrint": "Labyrinth",
    "Labyrint – AI:n tränar": "Labyrinth – the AI is training",
    "Luta brädet och för kulan från START till FINISH!": "Tilt the board and guide the ball from START to FINISH!",
    "Kunde inte ladda AI:n: {error}": "Could not load the AI: {error}",
    "Väntar på träningen\n{run} ...": "Waiting for the training\n{run} ...",
    "Tränad {steps} av {total} M steg\n= ca {practice} övning": "Trained {steps} of {total} M steps\n= about {practice} of practice",
    "{hours} timmars": "{hours} hours",
    "{minutes} minuters": "{minutes} minutes",
    "Kulan står still.\nNytt försök!": "The ball has stopped.\nNew attempt!",
    "Kulan föll i hål {n}!": "The ball fell into hole {n}!",
    "I MÅL på {time} s!": "FINISHED in {time} s!",
    "\nTryck R för att spela igen.": "\nPress R to play again.",
    # dashboard.py
    "Labyrint – träningen": "Labyrinth – training",
    "Andel i mål": "Success rate",
    "hur ofta kulan når FINISH (%)": "how often the ball reaches FINISH (%)",
    "träning (hälften startar mitt i banan)": "training (half of the games start mid-level)",
    "prov från START": "tests from START",
    "Tid till mål": "Time to finish",
    "snittid på proven från START (sekunder)": "mean time of the tests from START (seconds)",
    "AI:n": "the AI",
    "Belöning per omgång": "Reward per game",
    "poängen AI:n försöker få så hög som möjligt": "the score the AI tries to maximise",
    "Förlust (value loss)": "Value loss",
    "hur fel AI:n gissar sin framtida belöning": "how wrong the AI's guess of its future reward is",
    "Utforskning": "Exploration",
    "slumpbrus på rattarna (std) - minskar när den blir säker": "random noise on the knobs (std) - shrinks as it gets confident",
    "Förutsägelse": "Prediction",
    "explained variance - 1 = förutser belöningen perfekt": "explained variance - 1 = predicts the reward perfectly",
    "miljoner träningssteg": "million training steps",
    "autopiloten ({seconds} s)": "autopilot ({seconds} s)",
    "Väntar på träningen i {run} ...": "Waiting for the training in {run} ...",
    "bana: {level}": "level: {level}",
    "{steps} av {total} M steg ({percent} %)": "{steps} of {total} M steps ({percent} %)",
    "{fps} steg/s": "{fps} steps/s",
    "KLAR": "DONE",
    "ingen ny data - har träningen stannat?": "no new data - has the training stopped?",
}

_language = "sv"


def set_language(language: str) -> None:
    global _language
    if language not in LANGUAGES:
        raise ValueError(f"unknown language {language!r}, choose one of {LANGUAGES}")
    _language = language


def get_language() -> str:
    return _language


def t(text: str) -> str:
    """The text in the current language."""
    return EN.get(text, text) if _language == "en" else text


def number(x: float, decimals: int = 1) -> str:
    """A number formatted for the current language: 8 967,5 in Swedish, 8,967.5 in English."""
    s = f"{x:,.{decimals}f}"
    return s if _language == "en" else s.replace(",", " ").replace(".", ",")
