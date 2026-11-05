"""Tests for KeyMap load hardening: corrupt-file recovery and binding validation."""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import pytest

from ytm_player.config.keymap import Action, KeyMap, MatchResult


def test_corrupt_keymap_falls_back_to_defaults(tmp_path, caplog):
    """A malformed keymap.toml must not crash the app: back up + use defaults."""
    path = tmp_path / "keymap.toml"
    path.write_text("this is not valid toml === [[[", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        km = KeyMap.load(path)

    # Usable defaults are loaded.
    assert km.bindings
    assert km.match(("space",)) == (MatchResult.EXACT, Action.PLAY_PAUSE)
    # The broken file is backed up and moved out of the way.
    assert path.with_suffix(".toml.bak").exists()
    assert not path.exists()
    assert any("keymap" in r.getMessage().lower() for r in caplog.records)


def test_invalid_binding_value_is_skipped(tmp_path, caplog):
    """A non-str/non-list binding value logs a warning and is skipped, no crash."""
    path = tmp_path / "keymap.toml"
    path.write_text("[pages]\nlibrary = 5\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        km = KeyMap.load(path)

    # No crash; the default library binding remains intact (line skipped).
    assert ("g", "l") in km.get_keys_for_action(Action.LIBRARY)
    assert any("library" in r.getMessage() for r in caplog.records)


def test_list_with_non_str_element_is_skipped(tmp_path, caplog):
    """A list containing a non-str element is not a valid binding — skip + warn."""
    path = tmp_path / "keymap.toml"
    path.write_text('[pages]\nlibrary = ["g x", 7]\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        km = KeyMap.load(path)

    assert ("g", "l") in km.get_keys_for_action(Action.LIBRARY)
    assert any("library" in r.getMessage() for r in caplog.records)


def test_unknown_action_logs_warning(tmp_path, caplog):
    """An unknown action name is logged (not silently swallowed) and skipped."""
    path = tmp_path / "keymap.toml"
    path.write_text('[pages]\nnonexistent_action = "x"\n', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        km = KeyMap.load(path)

    assert km.bindings  # defaults still present
    assert any("nonexistent_action" in r.getMessage() for r in caplog.records)


def test_valid_override_replaces_default(tmp_path):
    """A well-formed override still rebinds the action (no regression)."""
    path = tmp_path / "keymap.toml"
    path.write_text('[pages]\nlibrary = "g x"\n', encoding="utf-8")

    km = KeyMap.load(path)

    assert km.match(("g", "x")) == (MatchResult.EXACT, Action.LIBRARY)
    assert km.match(("g", "l")) == (MatchResult.NO_MATCH, None)


def test_missing_file_uses_defaults(tmp_path):
    """No keymap.toml on disk → defaults, no backup, no crash."""
    path = tmp_path / "keymap.toml"
    km = KeyMap.load(path)
    assert km.match(("space",)) == (MatchResult.EXACT, Action.PLAY_PAUSE)
    assert not path.with_suffix(".toml.bak").exists()


def test_play_next_default_binding():
    """PLAY_NEXT binds to X / C-x by default (adjacent to add_to_queue's Z / C-z)."""
    km = KeyMap()
    km._load_defaults()
    assert km.match(("X",)) == (MatchResult.EXACT, Action.PLAY_NEXT)
    assert km.match(("C-x",)) == (MatchResult.EXACT, Action.PLAY_NEXT)


def test_removed_actions_are_gone():
    """CONTEXT_ACTIONS / SELECTED_ACTIONS / CLOSE_POPUP were retired — no bindings."""
    km = KeyMap()
    km._load_defaults()
    # The Action members no longer exist.
    for name in ("CONTEXT_ACTIONS", "SELECTED_ACTIONS", "CLOSE_POPUP"):
        assert not hasattr(Action, name)
    # Their old default keys resolve to nothing now.
    assert km.match(("g", "A")) == (MatchResult.NO_MATCH, None)
    assert km.match(("g", "a")) == (MatchResult.NO_MATCH, None)
    assert km.match(("C-space",)) == (MatchResult.NO_MATCH, None)
    # Escape is bound again — to clearing marks, not to the retired CLOSE_POPUP.
    assert km.match(("escape",)) == (MatchResult.EXACT, Action.MARK_CLEAR)


# ── native vs fallback parity ────────────────────────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def test_empty_sequence_is_pending_with_defaults():
    """() is a proper prefix of every binding, so it must stay PENDING."""
    km = KeyMap()
    km._load_defaults()
    assert km.match(()) == (MatchResult.PENDING, None)


def test_exact_match_beats_pending_prefix():
    """A sequence that is itself bound resolves EXACT even when it also
    prefixes a longer binding (exact lookup runs before the prefix scan)."""
    km = KeyMap()
    km._load_defaults()
    # ("d",) is only a proper prefix of the delete_item binding ("d", "d").
    assert km.match(("d",)) == (MatchResult.PENDING, None)
    # Bind ("d",) itself — exact must now win over the longer candidate.
    km.bindings[("d",)] = Action.DELETE_ITEM
    km._native_matcher = None  # honor the direct-mutation contract
    assert km.match(("d",)) == (MatchResult.EXACT, Action.DELETE_ITEM)
    assert km.match(("d", "d")) == (MatchResult.EXACT, Action.DELETE_ITEM)


# Child-interpreter program: builds the fallback KeyMap (native extension
# import blocked) and prints "result action" for one sequence.
_CHILD_MATCH_PROG = """
import json
import sys
sys.modules['ytm_player.utils._ytm_player_native'] = None
from pathlib import Path
from ytm_player.config.keymap import KeyMap
path, seq = json.loads(sys.argv[1])
if path is None:
    km = KeyMap()
    km._load_defaults()
else:
    km = KeyMap.load(Path(path))
result, action = km.match(tuple(seq))
print(result.value, action.value if action is not None else '-')
"""


def _fallback_match(path: str | None, seq: list[str]) -> str:
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_MATCH_PROG, json.dumps([path, seq])],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _match_repr(matched: tuple[MatchResult, Action | None]) -> str:
    result, action = matched
    return f"{result.value} {action.value if action is not None else '-'}"


# Default bindings: exact single keys, aliases, case sensitivity, exact
# multi-key sequences, pending prefixes, dead-end sequences, the empty
# sequence, and a sequence longer than any binding.
_DEFAULTS_SEQS: list[list[str]] = [
    [],
    ["space"],
    ["j"],
    ["down"],
    ["G"],
    ["g"],
    ["g", "g"],
    ["g", "l"],
    ["g", "space"],
    ["g", "z"],
    ["C-w"],
    ["C-w", "h"],
    ["C-w", "x"],
    ["d"],
    ["d", "d"],
    ["delete"],
    ["s"],
    ["s", "t"],
    ["s", "T"],
    ["q"],
    ["Q"],
    ["escape"],
    ["tab"],
    ["g", "space", "x"],
    ["x"],
]

# Overrides exercised by _CUSTOM_KEYMAP_TOML below.
_CUSTOM_SEQS: list[list[str]] = [
    ["g", "x"],  # overridden binding wins
    ["g", "l"],  # default binding removed by the override
    ["g"],  # still a pending prefix (g g etc. remain)
    [],  # empty-string binding resolves EXACT for the empty sequence
    ["z"],  # queue's default key removed
    ["?"],  # help unbound entirely via []
    ["C-h"],  # help's second default also gone
    ["j2"],  # alias added via a list-form binding
    ["j"],  # untouched default still bound
    ["y", "y"],  # unknown action ignored at load
    ["enter"],  # unrelated default unaffected
]

_CUSTOM_KEYMAP_TOML = (
    "[pages]\n"
    'library = "g x"\n'
    'queue = ""\n'
    "help = []\n"
    "\n"
    "[navigation]\n"
    'move_down = ["j", "down", "j2"]\n'
    "\n"
    "[actions]\n"
    'bogus_action = "y y"\n'
)


class TestKeyMapNativeParity:
    """When the Rust extension is built, KeyMap.match must answer exactly
    like the pure-Python fallback for every sequence shape.

    The fallback runs in a child interpreter with the extension import
    blocked, so both implementations are compared directly instead of
    trusting whichever one the test process happened to load.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("seq", _DEFAULTS_SEQS)
    def test_native_matches_fallback_defaults(self, seq):
        km = KeyMap()
        km._load_defaults()
        assert _match_repr(km.match(tuple(seq))) == _fallback_match(None, seq)
        # The native path really answered (not fallback vs fallback).
        assert type(km._native_matcher).__name__ == "KeyMatcher"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("seq", _CUSTOM_SEQS)
    def test_native_matches_fallback_custom_toml(self, tmp_path, seq):
        path = tmp_path / "keymap.toml"
        path.write_text(_CUSTOM_KEYMAP_TOML, encoding="utf-8")

        km = KeyMap.load(path)
        assert _match_repr(km.match(tuple(seq))) == _fallback_match(str(path), seq)
        # The native path really answered (not fallback vs fallback).
        assert type(km._native_matcher).__name__ == "KeyMatcher"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matcher_rebuilt_after_reload(self, tmp_path):
        """The matcher must be invalidated on every load — a stale native
        table would answer from the previous keymap.toml."""
        path = tmp_path / "keymap.toml"
        path.write_text('[pages]\nlibrary = "g x"\n', encoding="utf-8")

        km = KeyMap.load(path)
        assert km.match(("g", "x")) == (MatchResult.EXACT, Action.LIBRARY)
        assert type(km._native_matcher).__name__ == "KeyMatcher"

        path.write_text('[pages]\nlibrary = "g z"\n', encoding="utf-8")
        km2 = KeyMap.load(path)
        # Fresh load → matcher invalidated → answers from the NEW table.
        assert km2._native_matcher is None
        assert km2.match(("g", "x")) == (MatchResult.NO_MATCH, None)
        assert km2.match(("g", "z")) == (MatchResult.EXACT, Action.LIBRARY)

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_direct_mutation_contract(self):
        """Direct bindings writes need a manual _native_matcher reset —
        documented on the KeyMap docstring — and then must be reflected."""
        km = KeyMap()
        km._load_defaults()
        assert km.match(("y",)) == (MatchResult.NO_MATCH, None)

        km.bindings[("y", "y")] = Action.HELP
        km._native_matcher = None  # honor the direct-mutation contract
        assert km.match(("y",)) == (MatchResult.PENDING, None)
        assert km.match(("y", "y")) == (MatchResult.EXACT, Action.HELP)


# ── the load-plan engine: contracts + native/fallback parity ──────────

# Keymap TOMLs exercising every plan step and the ordering contract:
# valid string-form and list-form overrides, re-binding the same action
# from two sections (the second application removes the first's keys),
# the empty list (unbind) and the empty string (the empty sequence),
# whitespace-padded and multi-key strings, unknown actions, invalid
# binding values of several types, and non-dict top-level values.
_PLAN_KEYMAPS = {
    "overrides": (
        "[pages]\n"
        'library = "g x"\n'
        'queue = ""\n'
        "help = []\n"
        "\n"
        "[navigation]\n"
        'move_down = ["j", " down ", "j2"]\n'
        "\n"
        "[actions]\n"
        'bogus_action = "y y"\n'
    ),
    "rebinding_across_sections": (
        '[navigation]\nmove_down = "j x"\n[more_navigation]\nmove_down = "j y"\nmove_up = "k y"\n'
    ),
    "invalid_values": (
        "[pages]\n"
        "library = 5\n"
        'search = ["ok", 7]\n'
        "queue = true\n"
        "help = { not = 'a list' }\n"
        "\n"
        "[actions]\n"
        'like_toggle = "l"\n'
    ),
    "non_dict_section": ('standalone = 5\nanother = "text"\n\n[pages]\nlibrary = "g z"\n'),
    "whitespace_shapes": (
        '[navigation]\nmove_down = "  j  "\nmove_up = "k\\tt"\ngo_top = " g  g "\n'
    ),
    "empty": "",
}

_PLAN_CHILD_PROG = """
import json
import sys
from pathlib import Path

sys.modules['ytm_player.utils._ytm_player_native'] = None

import logging

logging.disable(logging.CRITICAL)

from ytm_player.config.keymap import KeyMap

payload = json.loads(sys.argv[1])
path = Path(payload["path"])
km = KeyMap.load(path)
out = {
    "bindings": sorted(
        [list(seq), action.value] for seq, action in km.bindings.items()
    ),
    "order": [list(seq) for seq in km.bindings],
    "warnings": payload.get("warnings", []),
}
json.dump(out, sys.stdout)
"""


_WARNINGS_CHILD_PROG = """
import json
import sys
from pathlib import Path

sys.modules['ytm_player.utils._ytm_player_native'] = None

import logging

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="%(message)s")

from ytm_player.config.keymap import KeyMap

payload = json.loads(sys.argv[1])
km = KeyMap.load(Path(payload["path"]))
"""


class TestKeyMapLoadPlanEngine:
    """The plan engine behind _load_from_dict, on whichever backend is
    loaded, plus the child-interpreter parity matrix when the extension
    is built."""

    def test_plan_steps_classify_every_entry(self, tmp_path, caplog):
        """The plan carries the unknown/invalid/bind classification in
        section order — the replay's warnings and applications follow
        it."""
        import logging

        from ytm_player.config.keymap import _keymap_load_plan

        data = {
            "pages": {
                "library": 5,
                "search": ["ok", 7],
                "queue": True,
                "help": {"not": "a list"},
            },
            "actions": {"like_toggle": "l"},
        }
        with caplog.at_level(logging.WARNING):
            plan = _keymap_load_plan(data)
        kinds = [step[0] for step in plan]
        assert kinds == ["invalid", "invalid", "invalid", "invalid", "bind"]
        # The names and type names ride along for the warnings.
        assert plan[0][1] == "library" and plan[0][2] == "int"
        assert plan[1][1] == "search" and plan[1][2] == "list"
        assert plan[3][1] == "help" and plan[3][2] == "dict"

    def test_rebinding_across_sections_removes_earlier_keys(self, tmp_path):
        """Two sections binding the same action: the second application
        removes the first's sequences — the reference's per-application
        _remove_action, preserved by the plan replay."""
        path = tmp_path / "keymap.toml"
        path.write_text(_PLAN_KEYMAPS["rebinding_across_sections"], encoding="utf-8")

        km = KeyMap.load(path)

        assert km.match(("j", "x")) == (MatchResult.NO_MATCH, None)
        assert km.match(("j", "y")) == (MatchResult.EXACT, Action.MOVE_DOWN)
        assert km.match(("k", "y")) == (MatchResult.EXACT, Action.MOVE_UP)
        # Both re-bound actions lost their whole default sets ("k" is now
        # only a prefix of the new sequence; "up"/"C-p" were removed with
        # the rest), while unrelated defaults survive untouched.
        assert km.match(("k",)) == (MatchResult.PENDING, None)
        assert km.match(("up",)) == (MatchResult.NO_MATCH, None)
        assert km.match(("space",)) == (MatchResult.EXACT, Action.PLAY_PAUSE)
        assert km.match(("n",)) == (MatchResult.EXACT, Action.NEXT_TRACK)

    def test_whitespace_shapes_split_on_python_whitespace(self, tmp_path):
        """Key strings strip then split on CPython's whitespace set —
        padded strings bind their inner sequence, a tab separates two
        keys."""
        path = tmp_path / "keymap.toml"
        path.write_text(_PLAN_KEYMAPS["whitespace_shapes"], encoding="utf-8")

        km = KeyMap.load(path)

        assert km.match(("j",)) == (MatchResult.EXACT, Action.MOVE_DOWN)
        assert km.match(("k", "t")) == (MatchResult.EXACT, Action.MOVE_UP)
        assert km.match(("g", "g")) == (MatchResult.EXACT, Action.GO_TOP)

    def test_non_dict_sections_are_skipped(self, tmp_path):
        path = tmp_path / "keymap.toml"
        path.write_text(_PLAN_KEYMAPS["non_dict_section"], encoding="utf-8")

        km = KeyMap.load(path)

        assert km.match(("g", "z")) == (MatchResult.EXACT, Action.LIBRARY)
        assert km.match(("g", "l")) == (MatchResult.NO_MATCH, None)

    def test_empty_file_keeps_defaults(self, tmp_path):
        path = tmp_path / "keymap.toml"
        path.write_text(_PLAN_KEYMAPS["empty"], encoding="utf-8")

        km = KeyMap.load(path)

        assert km.match(("space",)) == (MatchResult.EXACT, Action.PLAY_PAUSE)
        assert km.match(("g", "g")) == (MatchResult.EXACT, Action.GO_TOP)

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_loaded_bindings_match_fallback(self, tmp_path):
        """Every keymap zoo entry loads to the identical bindings table
        (contents AND order) as the child-interpreter fallback."""
        import json

        for name, toml_text in _PLAN_KEYMAPS.items():
            path = tmp_path / f"{name}.toml"
            path.write_text(toml_text, encoding="utf-8")

            km = KeyMap.load(path)
            native = {
                "bindings": sorted([list(seq), a.value] for seq, a in km.bindings.items()),
                "order": [list(seq) for seq in km.bindings],
            }

            result = subprocess.run(
                [sys.executable, "-c", _PLAN_CHILD_PROG, json.dumps({"path": str(path)})],
                capture_output=True,
                text=True,
                check=True,
            )
            fallback = json.loads(result.stdout)

            assert native["bindings"] == fallback["bindings"], name
            assert native["order"] == fallback["order"], f"binding order differs for {name!r}"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_load_warnings_match_fallback(self, tmp_path, caplog):
        """The warnings the replay emits — message for message, in
        order — are the fallback's own."""
        import json
        import logging

        path = tmp_path / "keymap.toml"
        path.write_text(_PLAN_KEYMAPS["invalid_values"], encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="ytm_player.config.keymap"):
            KeyMap.load(path)
        native_warnings = [r.getMessage() for r in caplog.records]

        # The child mirrors its warnings to stdout through a
        # logging.StreamHandler, one message per line.
        result = subprocess.run(
            [sys.executable, "-c", _WARNINGS_CHILD_PROG, json.dumps({"path": str(path)})],
            capture_output=True,
            text=True,
            check=True,
        )
        fallback_warnings = [w for w in result.stdout.splitlines() if w]

        assert native_warnings == fallback_warnings

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_native_engine_really_answers(self):
        """The binding is live: the module's plan builder delegates to
        the native function when the extension is built."""
        from ytm_player.config.keymap import _keymap_load_plan
        from ytm_player.utils._ytm_player_native import keymap_load_plan

        assert _keymap_load_plan({}) == []
        # The native engine is reachable through the wrapper (it returns
        # the native plan for a non-trivial input).
        plan = _keymap_load_plan({"pages": {"library": "g x"}})
        assert [step[0] for step in plan] == ["bind"]
        assert plan[0][2] == [("g", "x")]
        # The module-level name is the wrapper; the native function is
        # what it calls.
        assert callable(keymap_load_plan)
