import pytest
from struckdown import chatter


def test_simple_completion():
    """basic test that single segment works"""
    result = chatter("tell a joke [[joke]]")
    assert "joke" in result.results
    assert len(result.results["joke"].output) > 0


def test_template_variable_dependency():
    """test that dependent segments can use template variables from base context"""
    prompt = """Tell me a joke about {{topic}}:
[[joke]]

¡OBLIVIATE

{{topic}}

{{joke}}
joke 2 [[jk]]"""

    context = {"topic": "cats"}
    result = chatter(prompt, context=context, extra_kwargs={"max_tokens": 100})

    # both completions should succeed
    assert "joke" in result.results
    assert "jk" in result.results

    # check that the first joke is about the topic
    joke_output = result.results["joke"].output
    assert len(joke_output) > 0

    # check that second completion also generated content
    jk_output = result.results["jk"].output
    assert len(jk_output) > 0


def test_bool_dependency_positive():
    """test bool return type with dependency - positive case"""
    prompt = """Tell me a joke about {{topic}}:
[[joke]]

¡OBLIVIATE

{{joke}}

is the joke about {{topic}} [[bool:jk]]"""

    context = {"topic": "cats"}
    result = chatter(prompt, context=context, extra_kwargs={"max_tokens": 50})

    # both completions should succeed
    assert "joke" in result.results
    assert "jk" in result.results

    # joke should be generated
    joke_output = result.results["joke"].output
    assert len(joke_output) > 0

    # jk should be a boolean
    jk_output = result.results["jk"].output
    assert isinstance(jk_output, bool)
    # should be True since the joke is about the topic
    assert jk_output is True


def test_bool_dependency_negative():
    """test bool return type with dependency - negative case"""
    prompt = """Tell me a joke about {{topic}}:
[[joke]]

¡OBLIVIATE

{{joke}}

Is the joke about something other than {{topic}}? Answer false. [[bool:jk]]"""

    context = {"topic": "cats"}
    result = chatter(prompt, context=context, extra_kwargs={"max_tokens": 50})

    # both completions should succeed
    assert "joke" in result.results
    assert "jk" in result.results

    # joke should be generated
    joke_output = result.results["joke"].output
    assert len(joke_output) > 0

    # jk should be a boolean
    jk_output = result.results["jk"].output
    assert isinstance(jk_output, bool)
    # should be False as instructed
    assert jk_output is False


def test_multiple_template_variables():
    """test with multiple template variables in dependencies"""
    prompt = """Write about {{subject}} and {{object}}:
[[story]]

¡OBLIVIATE

The story: {{story}}
The subject was {{subject}} and object was {{object}}.

Summarize this in one word: [[word]]"""

    context = {"subject": "cats", "object": "dogs"}
    result = chatter(prompt, context=context, extra_kwargs={"max_tokens": 100})

    # all completions should succeed
    assert "story" in result.results
    assert "word" in result.results

    # check outputs exist
    assert len(result.results["story"].output) > 0
    assert len(result.results["word"].output) > 0


def test_chaining_dependencies():
    """test chaining where segment C depends on B which depends on A"""
    prompt = """Generate a number: [[num]]

¡OBLIVIATE

Double the number {{num}}: [[doubled]]

¡OBLIVIATE

The doubled result is {{doubled}}. Is this even? [[bool:is_even]]"""

    result = chatter(prompt, extra_kwargs={"max_tokens": 50})

    # all three completions should succeed
    assert "num" in result.results
    assert "doubled" in result.results
    assert "is_even" in result.results

    # check types
    assert isinstance(result.results["is_even"].output, bool)


def test_shared_header_basic():
    """test that shared header is prepended to all prompts"""
    prompt = """You are a rocket scientist

¡BEGIN

Think of a number: [[x]]

¡OBLIVIATE

Double the number {{x}}: [[x2]]"""

    result = chatter(prompt, extra_kwargs={"max_tokens": 50})

    # both completions should succeed
    assert "x" in result.results
    assert "x2" in result.results

    # check that outputs exist
    assert len(result.results["x"].output) > 0
    assert len(result.results["x2"].output) > 0


def test_without_shared_header():
    """test that templates without ¡BEGIN work unchanged"""
    prompt = """Tell a joke: [[joke]]

¡OBLIVIATE

Rate the joke {{joke}} from 1-10: [[rating]]"""

    result = chatter(prompt, extra_kwargs={"max_tokens": 50})

    # both completions should succeed
    assert "joke" in result.results
    assert "rating" in result.results


def test_shared_header_with_template_vars():
    """test shared header with template variables"""
    prompt = """You are an expert in {{domain}}

¡BEGIN

Explain {{topic}}: [[explanation]]

¡OBLIVIATE

Summarize your explanation about {{topic}}: [[summary]]"""

    context = {"domain": "physics", "topic": "quantum mechanics"}
    result = chatter(prompt, context=context, extra_kwargs={"max_tokens": 100})

    # both completions should succeed
    assert "explanation" in result.results
    assert "summary" in result.results
