"""Tests for the Spotify-import popup's flow engines.

Six engines behind ``SpotifyImportPopup``'s import flow: the
disambiguation candidate row (``_candidate_row`` — ``_CandidateItem``'s
label line), the per-track result arm (``_result_symbol`` — the ✓/?/✗
verdict both workers render), the multi-mode setup cascade
(``_multi_part_verdict``), the URL-slot validation
(``_url_slot_verdict``), the summary counting (``_summary_stats``) and
the create-batch toast verdict (``_batch_outcome``). The pure-Python
references live in ``ui/popups/spotify_import.py``; when the Rust
extension is built the same engines run natively
(``_ytm_player_native.spotify_import_*``). The uncertain-match bucket
walk both workers end with routes through the service's
``_bucket_results`` engine (``_ytm_player_native.spotify_bucket_results``
natively) — pinned here as part of the same wiring.

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart. The wiring tests mount
the popup itself and check the flow routes every verdict — the setup
notifications, the URL-slot focus moves, the summary status line and
the final toast — through the engines.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from ytm_player.services.spotify_import import MatchResult, MatchType
from ytm_player.ui.popups.spotify_import import (
    SpotifyImportPopup,
    _batch_outcome,
    _candidate_row,
    _multi_part_verdict,
    _result_symbol,
    _summary_stats,
    _url_slot_verdict,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_MEMBERS = {"exact": MatchType.EXACT, "multiple": MatchType.MULTIPLE, "none": MatchType.NONE}


def _result(match_type: str, selected: dict | None = None) -> MatchResult:
    """A minimal MatchResult carrying the bucket and selection."""
    return MatchResult({"name": "t", "artist": "a"}, _MEMBERS[match_type], selected=selected)


# ── Candidate row ─────────────────────────────────────────────────────


class TestCandidateRow:
    def test_full_row_with_seconds_duration(self):
        row = _candidate_row(
            1,
            {
                "title": "Song One",
                "artists": [{"name": "Artist One", "id": "a1"}],
                "duration_seconds": 125,
            },
        )
        assert row == "  1. Song One — Artist One (2:05)"

    def test_display_duration_wins_over_seconds(self):
        row = _candidate_row(2, {"title": "T", "artist": "A", "duration": "3:41"})
        assert row == "  2. T — A (3:41)"

    def test_missing_title_reads_the_question_default(self):
        row = _candidate_row(1, {"artist": "A"})
        assert row == "  1. ? — A"

    def test_missing_artist_reads_unknown(self):
        row = _candidate_row(1, {"title": "T"})
        assert row == "  1. T — Unknown"

    def test_zero_seconds_renders_no_duration_part(self):
        row = _candidate_row(1, {"title": "T", "artist": "A", "duration_seconds": 0})
        assert row == "  1. T — A"

    def test_missing_duration_fields_render_no_part(self):
        row = _candidate_row(7, {"title": "T", "artist": "A"})
        assert row == "  7. T — A"

    def test_seconds_pad_below_ten(self):
        row = _candidate_row(1, {"title": "T", "artist": "A", "duration_seconds": 61})
        assert row == "  1. T — A (1:01)"

    def test_falsy_duration_falls_to_seconds(self):
        # A present-but-empty duration field is falsy — the seconds
        # fallback renders.
        row = _candidate_row(
            1, {"title": "T", "artist": "A", "duration": "", "duration_seconds": 65}
        )
        assert row == "  1. T — A (1:05)"

    def test_artists_list_joins_before_the_duration(self):
        row = _candidate_row(
            1,
            {"title": "T", "artists": [{"name": "A", "id": "1"}, {"name": "B", "id": "2"}]},
        )
        assert row == "  1. T — A, B"


# ── Result symbol ────────────────────────────────────────────────────


class TestResultSymbol:
    def test_exact_answers_the_check_arm(self):
        assert _result_symbol(MatchType.EXACT) == ("✓", "success")

    def test_multiple_answers_the_question_arm(self):
        assert _result_symbol(MatchType.MULTIPLE) == ("?", "warning")

    def test_none_answers_the_cross_arm(self):
        assert _result_symbol(MatchType.NONE) == ("✗", "error")

    def test_non_member_objects_fall_to_the_cross_arm(self):
        # The comparison is the reference's own `==` against the enum
        # members — a plain string, even the member's value, is no member.
        assert _result_symbol("exact") == ("✗", "error")
        assert _result_symbol(None) == ("✗", "error")


# ── Multi-part verdict ───────────────────────────────────────────────


class TestMultiPartVerdict:
    def test_empty_name_answers_the_name_gate(self):
        assert _multi_part_verdict("", "5") == ("name", 0)

    def test_valid_count_answers_ok_with_the_count(self):
        assert _multi_part_verdict("Combined", "5") == ("ok", 5)

    def test_non_numeric_count_answers_the_count_gate(self):
        assert _multi_part_verdict("Combined", "abc") == ("count", 0)
        assert _multi_part_verdict("Combined", "3.5") == ("count", 0)

    def test_out_of_range_counts_answer_the_range_gate(self):
        assert _multi_part_verdict("Combined", "1") == ("range", 0)
        assert _multi_part_verdict("Combined", "21") == ("range", 0)
        assert _multi_part_verdict("Combined", "-3") == ("range", 0)

    def test_the_bounds_are_inclusive(self):
        assert _multi_part_verdict("Combined", "2") == ("ok", 2)
        assert _multi_part_verdict("Combined", "20") == ("ok", 20)

    def test_whitespace_and_underscore_numbers_parse(self):
        # CPython's own int() — surrounding whitespace and underscores
        # both parse on either backend.
        assert _multi_part_verdict("Combined", " 4 ") == ("ok", 4)
        assert _multi_part_verdict("Combined", "1_0") == ("ok", 10)

    def test_huge_counts_answer_the_range_gate(self):
        # A bignum parses and reads out-of-range like the reference's
        # own comparison would.
        assert _multi_part_verdict("Combined", "9" * 25) == ("range", 0)

    def test_type_errors_escape_like_the_references_arm(self):
        # The reference catches only ValueError; int(None) raises
        # TypeError on both backends and escapes to the caller.
        with pytest.raises(TypeError):
            _multi_part_verdict("Combined", None)  # type: ignore[arg-type]


# ── URL-slot verdict ─────────────────────────────────────────────────


_VALID_URL = "https://open.spotify.com/playlist/abc123"
_VALID_ALBUM = "http://open.spotify.com/album/xyz789"


class TestUrlSlotVerdict:
    def test_all_valid_answers_none(self):
        assert _url_slot_verdict([_VALID_URL, _VALID_ALBUM]) is None

    def test_empty_list_answers_none(self):
        assert _url_slot_verdict([]) is None

    def test_first_invalid_slot_with_its_index(self):
        assert _url_slot_verdict([_VALID_URL, "not-a-url"]) == ("invalid", 1)

    def test_only_the_first_bad_slot_answers(self):
        assert _url_slot_verdict(["bogus", "also-bogus"]) == ("invalid", 0)

    def test_missing_slots_answer_missing(self):
        assert _url_slot_verdict([None]) == ("missing", 0)
        assert _url_slot_verdict([_VALID_URL, None, "bogus"]) == ("missing", 1)

    def test_empty_string_is_invalid(self):
        assert _url_slot_verdict([""]) == ("invalid", 0)

    def test_wrong_scheme_host_or_type_is_invalid(self):
        assert _url_slot_verdict(["HTTP://open.spotify.com/playlist/x"]) == ("invalid", 0)
        assert _url_slot_verdict(["https://open.spotify.com/track/x"]) == ("invalid", 0)
        assert _url_slot_verdict(["spotify:playlist:x"]) == ("invalid", 0)
        assert _url_slot_verdict(["https://open.spotify.com/playlist"]) == ("invalid", 0)

    def test_trailing_text_after_the_slash_still_matches(self):
        # re.match anchors at the start only — the ID and anything
        # after it are fine.
        assert _url_slot_verdict(["https://open.spotify.com/playlist/x?si=1"]) is None


# ── Summary stats ────────────────────────────────────────────────────


class TestSummaryStats:
    def test_empty_results_count_zero_everything(self):
        assert _summary_stats([]) == {
            "exact": 0,
            "fuzzy": 0,
            "not_found": 0,
            "total": 0,
            "matched": 0,
            "video_ids": [],
        }

    def test_each_bucket_counts_its_own_members(self):
        stats = _summary_stats(
            [
                _result("exact", {"video_id": "v1"}),
                _result("multiple"),
                _result("multiple"),
                _result("none"),
            ]
        )
        assert stats["exact"] == 1
        assert stats["fuzzy"] == 2
        assert stats["not_found"] == 1
        assert stats["total"] == 4
        assert stats["matched"] == 3

    def test_matched_is_exact_plus_fuzzy(self):
        stats = _summary_stats([_result("exact"), _result("none")])
        assert stats["matched"] == 1

    def test_video_ids_collect_the_selected_ids_only(self):
        stats = _summary_stats(
            [
                _result("exact", {"video_id": "aaa"}),
                _result("exact", {"videoId": "bbb"}),
                _result("none"),
            ]
        )
        assert stats["video_ids"] == ["aaa", "bbb"]

    def test_unselected_multiple_results_contribute_no_ids(self):
        assert _summary_stats([_result("multiple")])["video_ids"] == []

    def test_falsy_selections_still_read_their_empty_id(self):
        # The gate is `is not None`, not truthiness — an empty dict
        # selection reads its (empty) id and drops out at the filter.
        assert _summary_stats([_result("exact", {})])["video_ids"] == []

    def test_empty_ids_drop_out_of_the_collection(self):
        stats = _summary_stats(
            [
                _result("exact", {"video_id": ""}),
                _result("exact", {"video_id": "ccc"}),
            ]
        )
        assert stats["video_ids"] == ["ccc"]


# ── Batch outcome ────────────────────────────────────────────────────


class TestBatchOutcome:
    def test_all_success_answers_the_success_verdict(self):
        outcome = _batch_outcome(["success", "success"], [100, 50])
        assert outcome == {
            "verdict": "success",
            "kind": None,
            "added_total": 150,
            "failed_batches": 0,
            "batches": 2,
        }

    def test_all_failed_the_same_way_answers_single_with_the_kind(self):
        outcome = _batch_outcome(["auth_expired", "auth_expired", "auth_expired"], [100, 100, 5])
        assert outcome["verdict"] == "all_failed_single"
        assert outcome["kind"] == "auth_expired"
        assert outcome["added_total"] == 0
        assert outcome["failed_batches"] == 3
        assert outcome["batches"] == 3

    def test_all_failed_different_ways_answers_mixed(self):
        outcome = _batch_outcome(["auth_expired", "network"], [100, 50])
        assert outcome["verdict"] == "all_failed_mixed"
        assert outcome["kind"] is None
        assert outcome["failed_batches"] == 2

    def test_partial_answers_with_the_accepted_total(self):
        outcome = _batch_outcome(["success", "quota", "success"], [100, 100, 20])
        assert outcome["verdict"] == "partial"
        assert outcome["added_total"] == 120
        assert outcome["failed_batches"] == 1
        assert outcome["batches"] == 3

    def test_a_single_failed_batch_is_the_single_kind(self):
        outcome = _batch_outcome(["rejected"], [100])
        assert outcome["verdict"] == "all_failed_single"
        assert outcome["kind"] == "rejected"

    def test_the_first_failing_kind_rides_along(self):
        outcome = _batch_outcome(["timeout", "timeout"], [100, 50])
        assert outcome["kind"] == "timeout"

    def test_empty_batches_answer_the_success_arm(self):
        # The wrapper never calls the engine with no batches (its own
        # total>0 gate); the engine stays total and answers the
        # no-failures arm.
        outcome = _batch_outcome([], [])
        assert outcome["verdict"] == "success"
        assert outcome["batches"] == 0


# ── Flow wiring ──────────────────────────────────────────────────────


class _Host(App):
    """Minimal host exposing the ``ytmusic`` attribute the popup reads."""

    def __init__(self, ytmusic: object | None = None) -> None:
        super().__init__()
        self.ytmusic = ytmusic

    def compose(self) -> ComposeResult:
        yield Static("host")


class _FakeCreateYTMusic:
    """Stands in for the app's YTMusic service: a canned playlist id
    with canned per-batch add answers, popped in batch order."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.batches: list[list[str]] = []

    async def create_playlist(self, name: str, desc: str) -> tuple[str, str]:
        return "success", "pl-created-1"

    async def add_playlist_items(
        self, playlist_id: str, batch: list[str], duplicates: bool = True
    ) -> str:
        self.batches.append(list(batch))
        return self._answers.pop(0)


async def _mounted_popup(app: App) -> SpotifyImportPopup:
    popup = SpotifyImportPopup()
    await app.push_screen(popup)
    return popup


class TestMultiSetupWiring:
    async def test_empty_name_notifies_and_holds_the_setup(self, monkeypatch):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-count-input", Input).value = "5"

            popup._generate_url_inputs()
            await pilot.pause()

            popup.notify.assert_called_once()
            assert "combined playlist name" in popup.notify.call_args.args[0]
            assert popup.notify.call_args.kwargs.get("severity") == "warning"
            # The setup held: no URL inputs mounted, the phase unchanged.
            assert popup._phase == "multi_setup"
            assert not list(popup.query("#si-multi-urls Input"))

    async def test_non_numeric_count_notifies_and_holds(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-name-input", Input).value = "Combined"
            popup.query_one("#si-multi-count-input", Input).value = "abc"

            popup._generate_url_inputs()
            await pilot.pause()

            popup.notify.assert_called_once()
            assert "valid number" in popup.notify.call_args.args[0]
            assert popup._phase == "multi_setup"

    async def test_out_of_range_count_notifies_and_holds(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-name-input", Input).value = "Combined"
            popup.query_one("#si-multi-count-input", Input).value = "21"

            popup._generate_url_inputs()
            await pilot.pause()

            popup.notify.assert_called_once()
            assert "between 2 and 20" in popup.notify.call_args.args[0]
            assert popup._phase == "multi_setup"

    async def test_the_ok_verdict_mounts_the_count_inputs(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-name-input", Input).value = "Combined"
            popup.query_one("#si-multi-count-input", Input).value = "3"

            popup._generate_url_inputs()
            await pilot.pause()

            assert popup._phase == "multi_urls"
            assert popup._multi_name == "Combined"
            assert popup._multi_url_count == 3
            inputs = list(popup.query("#si-multi-urls Input"))
            assert [i.id for i in inputs] == [f"si-multi-url-{n}" for n in range(3)]


class TestMultiStartWiring:
    async def test_invalid_url_notifies_and_focuses_the_slot(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            started: list[str] = []

            def fake_run_worker(coro, **kwargs):
                started.append(kwargs.get("name"))
                coro.close()

            popup.run_worker = fake_run_worker  # type: ignore[method-assign]
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-name-input", Input).value = "Combined"
            popup.query_one("#si-multi-count-input", Input).value = "2"
            popup._generate_url_inputs()
            await pilot.pause()
            popup.query_one(
                "#si-multi-url-0", Input
            ).value = "https://open.spotify.com/playlist/aaa"
            popup.query_one("#si-multi-url-1", Input).value = "not-a-url"

            popup._start_multi_import()
            await pilot.pause()

            popup.notify.assert_called_once()
            assert "Invalid Spotify URL in slot 2" in popup.notify.call_args.args[0]
            assert popup.notify.call_args.kwargs.get("severity") == "warning"
            # The flow stopped before the worker.
            assert started == []
            assert popup._phase == "multi_urls"

    async def test_valid_urls_stage_the_worker(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            started: list[str] = []

            def fake_run_worker(coro, **kwargs):
                started.append(kwargs.get("name"))
                coro.close()

            popup.run_worker = fake_run_worker  # type: ignore[method-assign]
            popup._activate_tab("multi")
            await pilot.pause()
            popup.query_one("#si-multi-name-input", Input).value = "Combined"
            popup.query_one("#si-multi-count-input", Input).value = "2"
            popup._generate_url_inputs()
            await pilot.pause()
            popup.query_one(
                "#si-multi-url-0", Input
            ).value = "https://open.spotify.com/playlist/aaa"
            popup.query_one(
                "#si-multi-url-1", Input
            ).value = "https://open.spotify.com/playlist/bbb"

            popup._start_multi_import()
            await pilot.pause()

            assert started == ["spotify_multi_import"]
            assert popup._multi_urls == [
                "https://open.spotify.com/playlist/aaa",
                "https://open.spotify.com/playlist/bbb",
            ]
            assert popup._phase == "multi_progress"


class TestSummaryWiring:
    async def test_the_status_line_assembles_from_the_engines_counts(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup._results = [
                _result("exact", {"video_id": "vid1"}),
                _result("multiple"),
                _result("none"),
            ]
            popup._playlist_name = "Source List"

            popup._show_summary()
            await pilot.pause()

            assert popup._phase == "summary"
            assert popup._video_ids == ["vid1"]
            status = popup.query_one("#si-status", Static)
            assert "2 matched, 1 skipped/not found (3 total)" in str(status._Static__content)
            name_input = popup.query_one("#si-name-input", Input)
            assert name_input.value == "Source List"
            assert name_input.has_class("visible")

    async def test_nothing_matched_answers_the_nothing_to_create_arm(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup._results = [_result("none"), _result("none")]

            popup._show_summary()
            await pilot.pause()

            assert popup._phase == "summary"
            assert popup._video_ids == []
            summary = popup.query_one("#si-summary", Static)
            assert "No tracks matched" in str(summary._Static__content)
            assert summary.has_class("visible")
            assert not popup.query_one("#si-name-input", Input).has_class("visible")

    async def test_an_empty_selection_id_collects_nothing(self):
        app = _Host()
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup._results = [_result("exact", {})]

            popup._show_summary()
            await pilot.pause()

            assert popup._video_ids == []


class TestCreateWiring:
    async def _run_create(self, answers: list[str]) -> tuple[MagicMock, SpotifyImportPopup, list]:
        app = _Host(_FakeCreateYTMusic(answers))
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            dismissed: list = []
            popup.dismiss = lambda result=None: dismissed.append(result)
            popup._video_ids = [f"v{i}" for i in range(150)]  # two batches: 100 + 50
            popup._playlist_name = "Source List"

            await popup._do_create("NewName")
            await pilot.pause()
            return popup.notify, popup, dismissed

    async def test_all_batches_accepted_toasts_the_success_arm(self):
        notify, popup, dismissed = await self._run_create(["success", "success"])
        notify.assert_called_once()
        assert notify.call_args.args[0] == "Created 'NewName' with 150 tracks"
        assert notify.call_args.kwargs.get("severity") == "information"
        assert dismissed == ["pl-created-1"]

    async def test_all_batches_rejected_the_same_way_toasts_the_suffix(self):
        notify, popup, dismissed = await self._run_create(["auth_expired", "auth_expired"])
        notify.assert_called_once()
        assert notify.call_args.args[0] == (
            "Created 'NewName' but couldn't add any tracks — "
            "session expired, run `ytm setup` to sign in again"
        )
        assert notify.call_args.kwargs.get("severity") == "error"

    async def test_all_batches_rejected_mixed_toasts_the_generic_arm(self):
        notify, popup, dismissed = await self._run_create(["auth_expired", "network"])
        notify.assert_called_once()
        assert notify.call_args.args[0] == (
            "Created 'NewName' but failed to add any tracks (2 batch(es) rejected)"
        )
        assert notify.call_args.kwargs.get("severity") == "error"

    async def test_some_batches_failed_toasts_the_partial_arm(self):
        notify, popup, dismissed = await self._run_create(["success", "auth_expired"])
        notify.assert_called_once()
        assert notify.call_args.args[0] == (
            "Created 'NewName' with partial track set: 100/150 added (1 batch(es) failed)"
        )
        assert notify.call_args.kwargs.get("severity") == "warning"

    async def test_the_batches_carry_the_full_id_list_in_order(self):
        app = _Host(_FakeCreateYTMusic(["success", "success"]))
        async with app.run_test() as pilot:
            popup = await _mounted_popup(app)
            await pilot.pause()
            popup.notify = MagicMock()
            popup.dismiss = lambda result=None: None
            ids = [f"v{i}" for i in range(150)]
            popup._video_ids = ids
            popup._playlist_name = "Source List"

            await popup._do_create("NewName")
            await pilot.pause()

            service = app.ytmusic
            assert [len(b) for b in service.batches] == [100, 50]  # type: ignore[attr-defined]
            assert service.batches[0] == ids[:100]  # type: ignore[attr-defined]


# ── Engine binding pins ──────────────────────────────────────────────


class TestEngineBindingPins:
    """The popup's class body routes every flow decision through the
    module-level engine names — the names the native tail swaps — so
    the bindings can never go stale, and the inline walks the engines
    replaced are gone."""

    def test_the_workers_and_views_route_through_the_engines(self):
        source = inspect.getsource(SpotifyImportPopup)
        # Both import workers' per-track display arm.
        assert source.count("_result_symbol(result.match_type)") == 2
        # Both workers' uncertain bucket walk.
        assert source.count("_bucket_results(") == 2
        # The setup cascade, the URL-slot pass, the summary and the
        # batch verdict.
        assert "_multi_part_verdict(name, count_str)" in source
        assert "_url_slot_verdict(values)" in source
        assert "_summary_stats(self._results)" in source
        assert "_batch_outcome(batch_results, batch_sizes)" in source

    def test_the_replaced_inline_walks_are_gone(self):
        source = inspect.getsource(SpotifyImportPopup)
        assert "sum(1 for" not in source, "the summary counting moved to the engine"
        assert "int(count_str)" not in source, "the count parse moved to the engine"
        assert "_URL_PATTERN.match" not in source, "the URL pass moved to the engine"
        assert "failure_kinds" not in source, "the batch accounting moved to the engine"
        assert "get_video_id" not in source, "the video-id collection moved to the engine"

    def test_the_candidate_rows_route_through_the_row_engine(self):
        from ytm_player.ui.popups.spotify_import import _CandidateItem

        source = inspect.getsource(_CandidateItem.compose)
        assert "_candidate_row(self._index, self.candidate)" in source

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_module_engines_are_the_native_ones(self):
        assert _candidate_row is _ytm_player_native.spotify_import_candidate_row
        assert _result_symbol is _ytm_player_native.spotify_import_result_symbol
        assert _multi_part_verdict is _ytm_player_native.spotify_import_multi_part_verdict
        assert _url_slot_verdict is _ytm_player_native.spotify_import_url_slot_verdict
        assert _summary_stats is _ytm_player_native.spotify_import_summary_stats
        assert _batch_outcome is _ytm_player_native.spotify_import_batch_outcome


# ── Native / fallback parity ────────────────────────────────────────
#
# When the Rust extension is built, the engines above ARE the native
# ones; each case below is also run in a child interpreter with the
# extension import blocked (pure-Python reference), and the two answers
# must agree exactly — rows, symbols, verdicts, counts, toast outcomes
# and error shapes included, JSON-normalised where the child's
# round-trip turns tuples into lists.

# (idx, candidate dict)
_PARITY_ROWS = [
    (
        1,
        {
            "title": "Song One",
            "artists": [{"name": "Artist One", "id": "a1"}],
            "duration_seconds": 125,
        },
    ),
    (2, {"title": "T", "artist": "A", "duration": "3:41"}),
    (1, {"artist": "A"}),
    (1, {"title": "T"}),
    (7, {"title": "T", "artist": "A"}),
    (1, {"title": "T", "artist": "A", "duration_seconds": 61}),
    (1, {"title": "T", "artist": "A", "duration": "", "duration_seconds": 65}),
    (
        3,
        {
            "title": "Ünïcode Ð",
            "artists": [{"name": "Ørchestra", "id": "o1"}],
            "duration_seconds": 6001,
        },
    ),
    (1, {"title": "T", "artist": "A", "duration": "bad", "duration_seconds": 125}),
]

# match_type value
_PARITY_SYMBOLS = ["exact", "multiple", "none", "bogus"]

# (name, count_str)
_PARITY_VERDICTS = [
    ("Combined", "5"),
    ("", "5"),
    ("Combined", "abc"),
    ("Combined", "3.5"),
    ("Combined", "1"),
    ("Combined", "21"),
    ("Combined", "2"),
    ("Combined", "20"),
    ("Combined", " 4 "),
    ("Combined", "1_0"),
    ("Combined", "9" * 25),
    ("Combined", "0x10"),
    ("Combined", "+7"),
    ("Combined", "٣"),
]

# list of value-or-null
_PARITY_URL_SLOTS = [
    [],
    ["https://open.spotify.com/playlist/abc123"],
    ["https://open.spotify.com/playlist/abc123", "not-a-url"],
    ["bogus", "also-bogus"],
    [None],
    ["https://open.spotify.com/playlist/x", None, "bogus"],
    [""],
    ["HTTP://open.spotify.com/playlist/x"],
    ["https://open.spotify.com/track/x"],
    ["spotify:playlist:x"],
    ["https://open.spotify.com/playlist"],
    ["http://open.spotify.com/album/x?si=1"],
    ["https://open.spotify.com/playlist/x extra text"],
]

# list of (match_type value, selected dict-or-null)
_PARITY_SUMMARIES = [
    [],
    [("exact", {"video_id": "v1"})],
    [("exact", {"video_id": "v1"}), ("multiple", None), ("multiple", None), ("none", None)],
    [("exact", {"video_id": "aaa"}), ("exact", {"videoId": "bbb"}), ("none", None)],
    [("multiple", None)],
    [("exact", {})],
    [("exact", {"video_id": ""}), ("exact", {"video_id": "ccc"})],
    [("none", None), ("none", None), ("multiple", None), ("exact", {"video_id": "z"})],
]

# (batch results, batch sizes)
_PARITY_BATCHES = [
    (["success", "success"], [100, 50]),
    (["auth_expired", "auth_expired", "auth_expired"], [100, 100, 5]),
    (["auth_expired", "network"], [100, 50]),
    (["success", "quota", "success"], [100, 100, 20]),
    (["rejected"], [100]),
    (["timeout", "timeout"], [100, 50]),
    ([], []),
    (["success"], [1]),
    (["duplicate", "duplicate", "duplicate"], [7, 8, 9]),
    (["success", "server_error", "server_error"], [10, 10, 10]),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.spotify_import import MatchResult, MatchType
from ytm_player.ui.popups.spotify_import import (
    _batch_outcome,
    _candidate_row,
    _multi_part_verdict,
    _result_symbol,
    _summary_stats,
    _url_slot_verdict,
)

MEMBERS = {
    "exact": MatchType.EXACT,
    "multiple": MatchType.MULTIPLE,
    "none": MatchType.NONE,
}


def result(spec):
    match_type, selected = spec
    return MatchResult({"name": "t", "artist": "a"}, MEMBERS[match_type], selected=selected)


def outcome(fn, *args):
    try:
        return {"ok": fn(*args)}
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)

rows = [outcome(_candidate_row, idx, cand) for idx, cand in payload["rows"]]
symbols = [outcome(_result_symbol, MEMBERS.get(value, value)) for value in payload["symbols"]]
verdicts = [
    {"ok": list(_multi_part_verdict(name, count_str))}
    for name, count_str in payload["verdicts"]
]
slots = [
    {"ok": None if verdict is None else list(verdict)}
    for verdict in (_url_slot_verdict(values) for values in payload["slots"])
]
summaries = [
    outcome(_summary_stats, [result(spec) for spec in specs]) for specs in payload["summaries"]
]
batches = [
    outcome(_batch_outcome, results, sizes) for results, sizes in payload["batches"]
]

json.dump(
    {
        "rows": rows,
        "symbols": symbols,
        "verdicts": verdicts,
        "slots": slots,
        "summaries": summaries,
        "batches": batches,
    },
    sys.stdout,
)
"""


def _normalize(value):
    """JSON-shape the native answers: the child's round-trip turns
    every tuple into a list, so the native tuples normalise the same
    way before the comparison."""
    if isinstance(value, tuple):
        return [_normalize(item) for item in value]
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _native_outcome(fn, *args):
    try:
        return {"ok": _normalize(fn(*args))}
    except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native flow engines must
    decide exactly like the pure-Python references — same rows, same
    symbols, same verdicts, same counts, same toast outcomes, same
    error shapes — for every case, so the two implementations cannot
    drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native_rows = [_native_outcome(_candidate_row, idx, cand) for idx, cand in _PARITY_ROWS]
        native_symbols = [
            _native_outcome(_result_symbol, _MEMBERS.get(value, value)) for value in _PARITY_SYMBOLS
        ]
        native_verdicts = [
            {"ok": list(_multi_part_verdict(name, count_str))}
            for name, count_str in _PARITY_VERDICTS
        ]
        native_slots = [
            {"ok": None if verdict is None else list(verdict)}
            for verdict in (_url_slot_verdict(values) for values in _PARITY_URL_SLOTS)
        ]
        native_summaries = [
            _native_outcome(_summary_stats, [_result(*spec) for spec in specs])
            for specs in _PARITY_SUMMARIES
        ]
        native_batches = [
            _native_outcome(_batch_outcome, results, sizes) for results, sizes in _PARITY_BATCHES
        ]

        payload = json.dumps(
            {
                "rows": _PARITY_ROWS,
                "symbols": _PARITY_SYMBOLS,
                "verdicts": _PARITY_VERDICTS,
                "slots": _PARITY_URL_SLOTS,
                "summaries": _PARITY_SUMMARIES,
                "batches": _PARITY_BATCHES,
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        for case, native, ref in zip(_PARITY_ROWS, native_rows, fallback["rows"]):
            assert native == ref, (
                f"native and fallback candidate rows differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
        for case, native, ref in zip(_PARITY_SYMBOLS, native_symbols, fallback["symbols"]):
            assert native == ref, (
                f"native and fallback result symbols differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
        for case, native, ref in zip(_PARITY_VERDICTS, native_verdicts, fallback["verdicts"]):
            assert native == ref, (
                f"native and fallback multi-part verdicts differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
        for case, native, ref in zip(_PARITY_URL_SLOTS, native_slots, fallback["slots"]):
            assert native == ref, (
                f"native and fallback URL-slot verdicts differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
        for case, native, ref in zip(_PARITY_SUMMARIES, native_summaries, fallback["summaries"]):
            assert native == ref, (
                f"native and fallback summary stats differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
        for case, native, ref in zip(_PARITY_BATCHES, native_batches, fallback["batches"]):
            assert native == ref, (
                f"native and fallback batch outcomes differ for {case!r}:\n"
                f"native:   {native!r}\nfallback: {ref!r}"
            )
