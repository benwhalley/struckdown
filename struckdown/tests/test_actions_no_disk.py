"""Actions take data from the context; none reads a path a template names.

And what may run in remote mode is read from each action's registration,
by the same parser that runs the template, never by matching its name.
"""

import pytest

import struckdown as sd
from struckdown.actions import Actions, ActionNotAllowed
from struckdown.actions.history import history_action, messages_from_text
from struckdown.playground.core import actions_in, check_disallowed_actions


def test_history_comes_from_the_context():
    msgs = messages_from_text("hello\nhi there\nhow are you")
    result = history_action({"_history_messages": msgs}, n=2)
    assert [m["content"] for m in result] == ["hi there", "how are you"]


def test_history_will_not_take_a_filename(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    # a filename is just an unknown filter now: ignored, and the file is not read
    with pytest.raises(ValueError, match="_history_messages"):
        history_action({}, filename=str(secret))
    with pytest.raises(ValueError, match="_history_messages"):
        history_action({"_history_file": str(secret)})


@pytest.mark.parametrize(
    "template",
    [
        "[[@fetch:p|url=x]]",
        '[[ @fetch:p|"http://169.254.169.254/"]]',
        "[[@fetch :p|u]]",
        "[[@fetch |url=x]]",
        "[[@fetch]]",
        "[[ @search:r|query=x]]",
    ],
)
def test_every_spelling_of_an_unsafe_action_is_found(template):
    assert actions_in(template) and check_disallowed_actions(template)


def test_an_unknown_action_runs_as_a_noop():
    assert actions_in("[[@nosuchaction:x|1]]") == {"noop"}
    assert check_disallowed_actions("[[@nosuchaction:x|1]]") == []


def test_safe_actions_pass():
    assert check_disallowed_actions("[[@set:x|1]] [[@timestamp:t]]") == []


def test_remote_policy_is_read_from_the_registration(monkeypatch):
    import struckdown.actions as actions

    monkeypatch.setattr(actions, "_remote_only", True)
    assert not Actions.is_allowed_remote("search")
    with pytest.raises(ActionNotAllowed):
        sd.complete("[[ @search:r|query=x]]", {})
    # a safe action still runs
    result = sd.complete("[[@set:x|1]]", {})
    assert "x" in result.results
