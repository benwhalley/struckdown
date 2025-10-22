"""Test block comments in struckdown templates

Block comments use the syntax: <!-- ... -->
They are completely dropped from the template before any other processing.
"""

import pytest
from struckdown.parsing import parse_syntax


def test_simple_block_comment():
    """Block comment should be completely ignored"""
    template = """
    <!--This is a comment-->

    Hello {{name}}

    [[response]]
    """

    result = parse_syntax(template)
    assert len(result) == 1
    section = result[0]
    prompt_text = section['response'].text

    # comment content should not appear in prompt
    assert 'This is a comment' not in prompt_text
    assert '{{name}}' in prompt_text


def test_block_comment_with_special_syntax():
    """Block comments can contain completions, actions, variables without affecting template"""
    template = """
    <!--
    This comment contains:
    - A completion: [[pick:ignored|a,b,c]]
    - A variable: {{ignored_var}}
    - An action: [[@evidence:ignored_data]]
    -->

    What is {{actual_var}}?

    [[actual_completion]]
    """

    result = parse_syntax(template)
    section = result[0]
    prompt_text = section['response'].text

    # block comment content should not leak
    assert 'ignored' not in prompt_text
    assert '{{ignored_var}}' not in prompt_text
    assert '[[pick:' not in prompt_text

    # actual content should be present
    assert '{{actual_var}}' in prompt_text


def test_multiple_block_comments():
    """Multiple block comments in same template"""
    template = """
    <!--First comment-->

    Some text

    <!--Second comment-->

    More text

    [[response]]
    """

    result = parse_syntax(template)
    section = result[0]
    prompt_text = section['response'].text

    assert 'First comment' not in prompt_text
    assert 'Second comment' not in prompt_text
    assert 'Some text' in prompt_text
    assert 'More text' in prompt_text


def test_block_comment_at_start():
    """Block comment at the very start of template"""
    template = """<!--
Simple Chain Example
Shows chaining completions with obliviate
-->

Write a joke about {{topic}}

[[joke]]"""

    result = parse_syntax(template)
    section = result[0]
    prompt_text = section['response'].text

    assert 'Simple Chain Example' not in prompt_text
    assert '{{topic}}' in prompt_text


def test_block_comment_with_obliviate():
    """Block comments work correctly with OBLIVIATE sections"""
    template = """
    <!--First section docs-->

    First prompt

    [[first]]

    ¡OBLIVIATE

    <!--Second section docs-->

    Second prompt

    [[second]]
    """

    result = parse_syntax(template)
    assert len(result) == 2

    # both sections should be clean
    # completions use variable names 'first' and 'second' as specified
    first_prompt = result[0]['response'].text
    second_prompt = result[1]['response'].text

    assert 'First section docs' not in first_prompt
    assert 'Second section docs' not in second_prompt
    assert 'First prompt' in first_prompt
    assert 'Second prompt' in second_prompt


def test_empty_block_comment():
    """Empty block comments should not cause errors"""
    template = """
    <!---->

    Hello world

    [[response]]
    """

    result = parse_syntax(template)
    assert len(result) == 1


def test_multiline_block_comment():
    """Block comments can span multiple lines"""
    template = """
    <!--
    Line 1
    Line 2
    Line 3
    With {{variables}} and [[completions]]
    -->

    Actual content

    [[response]]
    """

    result = parse_syntax(template)
    section = result[0]
    prompt_text = section['response'].text

    assert 'Line 1' not in prompt_text
    assert 'Line 2' not in prompt_text
    assert 'Line 3' not in prompt_text
    assert 'Actual content' in prompt_text
