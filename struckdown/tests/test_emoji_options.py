"""Emoji as unquoted pick options.

The EMOJI terminal took single code points from two ranges, so an emoji built from
a sequence -- a variation selector, a zero-width joiner, a flag -- or from any
other block failed to parse, and the whole template with it.
"""

import pytest

from struckdown.parsing import parse_slot_body, parse_syntax

EMOJI = [
    "👍",  # plain, in the original range
    "❤️",  # base + variation selector U+FE0F
    "❤‍🔥",  # ZWJ sequence
    "👨‍💻",  # ZWJ sequence
    "🤷‍♂️",  # ZWJ + gender sign + variation selector
    "👍🏽",  # skin tone modifier
    "🇬🇧",  # flag: regional indicator pair
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿",  # subdivision flag: tag characters
    "🆒",  # enclosed alphanumeric supplement
    "⭐",  # misc symbols and arrows
    "⌚",  # misc technical
    "👍👍",  # a run of emoji is one option
]


def options(template: str) -> list[str]:
    return [o.value for o in parse_syntax(template)[0]["r"].options]


@pytest.mark.parametrize("emoji", EMOJI)
def test_emoji_parses_as_an_option(emoji):
    assert options(f"Pick [[pick:r|yes,{emoji},no]]") == ["yes", emoji, "no"]


@pytest.mark.parametrize("emoji", EMOJI)
def test_emoji_parses_in_a_slot_body(emoji):
    """The slot-body grammar parses options rendered from variables."""
    assert parse_slot_body(f"pick:r|yes,{emoji},no")["options"][1].value == emoji


def test_every_emoji_in_one_list():
    assert options(f"Pick [[pick:r|{','.join(EMOJI)}]]") == EMOJI


def test_quoted_emoji_still_parse():
    assert options('Pick [[pick:r|"❤️","👍"]]') == ["❤️", "👍"]
