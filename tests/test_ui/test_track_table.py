"""Tests for TrackTable widget."""

from __future__ import annotations

from textual.app import App, ComposeResult

from ytm_player.config.keymap import Action
from ytm_player.ui.widgets.track_table import TrackTable


class _Host(App):
    """Minimal host that provides the theme variables TrackTable's CSS needs."""

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables


def _column_keys(table: TrackTable) -> set[str]:
    return {c.value for c in table.columns if c.value is not None}


async def test_load_tracks_immediately_after_construction():
    """Regression: see context._build_artist nested-mount path.

    Before option C (column setup moved from on_mount to __init__), a
    caller that constructed a TrackTable and called load_tracks
    synchronously — before on_mount fired — hit add_row with 0 columns
    and raised "More values provided than there are columns".

    This test reproduces that scenario: load_tracks runs during compose,
    so on_mount has not yet fired. With option C in place, columns exist
    at construction time and load_tracks succeeds.
    """

    captured: dict[str, int] = {}

    class _LoadDuringCompose(_Host):
        def compose(self) -> ComposeResult:
            table = TrackTable(show_index=True, show_album=False)
            table.load_tracks(
                [
                    {
                        "video_id": "abc",
                        "title": "Test Track",
                        "artist": "Test Artist",
                        "duration": 60,
                    }
                ]
            )
            captured["count"] = table.track_count
            yield table

    app = _LoadDuringCompose()
    async with app.run_test():
        assert captured["count"] == 1


async def test_columns_match_show_album_show_index_flags():
    """Column set tracks the show_album / show_index flags from __init__."""

    captured: dict[str, set[str]] = {}

    class _CaptureColumns(_Host):
        def compose(self) -> ComposeResult:
            no_album_no_index = TrackTable(show_index=False, show_album=False)
            captured["minimal"] = _column_keys(no_album_no_index)

            full = TrackTable(show_index=True, show_album=True)
            captured["full"] = _column_keys(full)

            yield no_album_no_index
            yield full

    app = _CaptureColumns()
    async with app.run_test():
        assert captured["minimal"] == {"mark", "title", "artist", "duration"}
        assert captured["full"] == {"mark", "index", "title", "artist", "album", "duration"}


async def test_selected_original_index_maps_through_sort_and_filter():
    """cursor_row is a visible-row index; selected_original_index must
    return the load-order index (what queue mutations need)."""

    tracks = [
        {"video_id": f"t{i}", "title": title, "artist": "A", "duration": 60}
        for i, title in enumerate(["d", "c", "b", "a"])
    ]

    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_index=True, show_album=False)

    app = _WithTable()
    async with app.run_test():
        table = app.query_one(TrackTable)
        table.load_tracks(tracks)
        table.move_cursor(row=0)
        assert table.selected_original_index == 0

        table.sort_by("title")  # visible: a,b,c,d = original 3,2,1,0
        table.move_cursor(row=0)
        assert table.selected_original_index == 3

        # Bypass apply_filter's debounce timer for determinism.
        table._filter_text = "b"
        table._execute_filter()  # visible: only "b" (original index 2)
        table.move_cursor(row=0)
        assert table.selected_original_index == 2

        table._filter_text = "no-match"
        table._execute_filter()
        assert table.selected_original_index is None


async def test_apply_filter_keeps_filter_active_truthful():
    """load_tracks resets _filter_active; a programmatic apply_filter (the
    background-refresh reapply path) must restore it so append_tracks keeps
    honouring the active filter."""

    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_index=True, show_album=False)

    app = _WithTable()
    async with app.run_test():
        table = app.query_one(TrackTable)
        table.load_tracks([{"video_id": "t1", "title": "a", "artist": "A", "duration": 60}])
        assert table._filter_active is False
        table.apply_filter("a")
        assert table._filter_active is True
        table.apply_filter("")
        assert table._filter_active is False


def _titled_tracks() -> list[dict]:
    return [
        {"video_id": f"t{i}", "title": title, "artist": "A", "duration": 60}
        for i, title in enumerate(["d", "c", "b", "a"])
    ]


async def test_clear_sort_returns_to_load_order_keeping_marks_filter_and_cursor():
    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_index=True, show_album=False)

    app = _WithTable()
    async with app.run_test():
        table = app.query_one(TrackTable)
        table.load_tracks(_titled_tracks(), keys=["k0", "k1", "k2", "k3"])
        table.sort_by("title")  # visible: a,b,c,d = original 3,2,1,0
        table.move_cursor(row=1)  # "b" = original 2
        await table.handle_action(Action.MARK_TOGGLE)
        assert [t["video_id"] for t in table.marked_tracks()] == ["t2"]

        table.clear_sort()

        assert [t["video_id"] for t in table.visible_tracks] == ["t0", "t1", "t2", "t3"]
        assert table.selected_original_index == 2
        assert table.occurrence_key(2) == "k2"
        assert [t["video_id"] for t in table.marked_tracks()] == ["t2"]
        assert table._sort_column is None

        # Filter survives too (bypass the debounce timer).
        table.sort_by("title")
        table._filter_text = "b"
        table._execute_filter()
        table.clear_sort()
        assert [t["video_id"] for t in table.visible_tracks] == ["t2"]
        assert table._filter_text == "b"


async def test_clear_sort_is_a_noop_when_unsorted():
    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_index=True, show_album=False)

    app = _WithTable()
    async with app.run_test():
        table = app.query_one(TrackTable)
        table.load_tracks(_titled_tracks())
        table.move_cursor(row=2)

        table.clear_sort()

        assert [t["video_id"] for t in table.visible_tracks] == ["t0", "t1", "t2", "t3"]
        assert table.selected_original_index == 2
        assert table.occurrence_key(2) is None  # loaded without keys


async def test_queue_entry_keys_is_off_unless_asked_for():
    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_album=False)
            yield TrackTable(show_album=False, queue_entry_keys=True, id="queue-like")

    app = _WithTable()
    async with app.run_test():
        plain, queue_like = app.query(TrackTable)
        assert plain.queue_entry_keys is False
        assert queue_like.queue_entry_keys is True


async def test_playing_highlight_styles_and_restores_through_the_row_engine():
    """The now-playing row's cells and the restored row's cells come from
    the ``_row_values`` engine: the styled row wraps the engine's values
    in bold Rich Text under the ▶ row label, the restored row keeps the
    engine's plain strings — its live mark glyph included — under a blank
    label, and clearing the playing track restores the last styled row."""

    from rich.text import Text

    from ytm_player.ui.widgets.track_table import _row_values

    tracks = [
        {"video_id": "v1", "title": "One", "artist": "A", "album": "Album A", "duration": 61},
        {"video_id": "v2", "title": "Two", "artist": "B", "album": "Album B", "duration": 120},
        {"video_id": "v3", "title": "Three", "artist": "C", "album": None, "duration": None},
    ]
    keys = ["mark", "index", "title", "artist", "album", "duration"]

    class _WithTable(_Host):
        def compose(self) -> ComposeResult:
            yield TrackTable(show_index=True, show_album=True)

    app = _WithTable()
    async with app.run_test():
        table = app.query_one(TrackTable)
        table.load_tracks(tracks)
        assert table._playing_index is None

        # Play the second track: bold engine cells + the ▶ label.
        table.set_playing("v2")
        assert table._playing_index == 1
        assert str(table.rows[table._row_keys[1]].label) == "▶"
        _, engine_cells = _row_values(table._tracks[1], 1, True, True, table._marked)
        for key, value in zip(keys, engine_cells):
            cell = table.get_cell(table._row_keys[1], key)
            assert isinstance(cell, Text)
            assert cell.plain == value
            assert "bold" in cell.style
        assert table.get_cell(table._row_keys[1], "duration").plain == "2:00"

        # Mark the playing row, then move on: the restore consults the
        # live marks set through the engine (the ✓ lands in the mark
        # cell) and answers plain strings under a blank label.
        table._set_marks({1})
        table.set_playing("v3")
        assert table._playing_index == 2
        _, restored = _row_values(table._tracks[1], 1, True, True, table._marked)
        for key, value in zip(keys, restored):
            cell = table.get_cell(table._row_keys[1], key)
            assert not isinstance(cell, Text)
            assert cell == value
        assert table.get_cell(table._row_keys[1], "mark") == "✓"
        assert table.get_cell(table._row_keys[1], "index") == "2"
        assert str(table.rows[table._row_keys[1]].label) == " "
        assert str(table.rows[table._row_keys[2]].label) == "▶"

        # The None-duration track renders the engine's --:-- placeholder.
        assert table.get_cell(table._row_keys[2], "duration").plain == "--:--"

        # Clearing the playing track restores the last styled row too.
        table.set_playing(None)
        assert table._playing_index is None
        _, cleared = _row_values(table._tracks[2], 2, True, True, table._marked)
        for key, value in zip(keys, cleared):
            cell = table.get_cell(table._row_keys[2], key)
            assert not isinstance(cell, Text)
            assert cell == value
        assert str(table.rows[table._row_keys[2]].label) == " "
