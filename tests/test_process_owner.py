"""The optional SDK process owner must preserve actual CLI success/failure codes."""
import pytest

from examples import _isaaclab_process as owner
from transformer_rl import cli


@pytest.mark.parametrize("outcome,expected", [(0, 0), (1, 1), (None, 0), ("system_exit", 2),
                                             ("exception", 1), ("interrupt", 130)])
def test_application_closes_after_cli_and_keeps_exit_code(monkeypatch, outcome, expected):
    monkeypatch.setattr(owner, "_apps", [])
    monkeypatch.setattr(owner, "_active", False)
    events = []

    class Application:
        def close(self, *, wait_for_replicator, exit_code):
            events.append(("close", wait_for_replicator, exit_code))

    def main():
        owner.register_app(Application())
        events.append("cli")
        if outcome == "system_exit":
            raise SystemExit(2)
        if outcome == "exception":
            raise RuntimeError("scripted failure")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        return outcome

    monkeypatch.setattr(cli, "main", main)
    assert owner.main() == expected
    assert events == ["cli", ("close", False, expected)]


def test_factory_cannot_register_without_explicit_process_owner(monkeypatch):
    monkeypatch.setattr(owner, "_active", False)
    with pytest.raises(RuntimeError, match="process"):
        owner.register_app(object())
