"""Tests for the auth session-record engines.

The trust machinery around auth.json and account.json runs as engines
with two interchangeable backends: the pure-Python transcriptions in
``services/auth.py`` and, when the Rust extension is built, the native
implementations bound at the module's tail. The engines:

``_parse_record`` — account.json bytes as a dict, or None when the bytes
are missing, malformed, another schema, or undecodable. The JSON walk
is CPython's own ``json.loads`` grammar (the ``01``/``1.``/``+1``
rejections, the digit-required fraction and exponent, the literals
null/true/false/NaN/Infinity/-Infinity prefix-matched so trailing junk
rejects at the delimiter, duplicate object keys keeping the first
position with the last value, ``\\uXXXX`` with surrogate pairing — a
high surrogate combines only with a low one, every other shape keeping
its lone surrogate — arbitrary-precision integers, floats through the
same correctly-rounded conversion, and the same RecursionError for a
payload nested past the interpreter's recursion budget). The schema
gate compares through CPython's own ``!=``, so a schema_version of 1.0
or True passes exactly like 1 does.

``_identity_from_record`` — the renewal gate: the ``verified`` field is
gated by identity with the False singleton (a 0 is NOT False and passes
the gate), the channel id must fullmatch the UC…22 pattern, and the
slot must be a digit string under CPython's own ``str.isdigit()`` —
Unicode digits included, so ``int('٣')`` keeps its value and an
isdigit-true, int-false spelling raises ``int()``'s own ValueError.

``_owner_of`` — the commit guard's snapshot: the auth bytes' SHA-256
and the record's revision, answered only when the record is truthy and
its ``auth_sha256`` compares equal to that hash.

``_vouch_failure`` — the stream-jar vouching decision tree in the
reference's exact order, the two legacy shapes accepted, every refusal
reason spelled as the reference spells it.

``_channel_id_from_account_menu`` — the defensive walk that extracts
the signed-in channel id from an account_menu response and fails
closed on every ambiguous shape.

``_serialize_headers`` / ``_candidate_payload`` — the one canonical
auth.json spelling (``json.dumps(..., ensure_ascii=True, indent=4,
sort_keys=True)``) and the candidate bytes for one browser slot (the
dict merge, the slot rendered through ``str()``).

``_build_record`` — the account.json assembly with its fixed key order
and the falsy-handle-to-None gate.

``_sha256`` — the hash primitive.

The unit tests below assert each engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case —
including the error shapes, where CPython's own messages must surface
identically — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

from ytm_player.services.auth import (
    _build_record,
    _candidate_payload,
    _channel_id_from_account_menu,
    _FileRead,
    _identity_from_record,
    _owner_of,
    _parse_record,
    _ProbedAccount,
    _serialize_headers,
    _sha256,
    _vouch_failure,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

ME_ID = "UCa1b2c3d4e5f6g7h8i9j0k1"
OTHER_ID = "UCz9y8x7w6v5u4t3s2r1q0p9"


def _record_bytes(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=True, indent=4, sort_keys=True).encode("utf-8")


class TestParseRecord:
    """account.json bytes as a dict — or None for every unusable shape."""

    def test_none_passthrough(self):
        assert _parse_record(None) is None

    def test_valid_record_round_trips(self):
        raw = _record_bytes(
            {
                "schema_version": 1,
                "channel_id": ME_ID,
                "x-goog-authuser": "0",
                "verified": True,
                "auth_sha256": "ab" * 32,
                "revision": "cd" * 16,
            }
        )
        assert _parse_record(raw) == json.loads(raw.decode("utf-8"))

    @pytest.mark.parametrize(
        "raw",
        [
            b"",  # empty
            b"nope",  # not JSON
            b"01",  # leading zero
            b"1.",  # digitless fraction
            b".5",  # digitless integer part
            b"+1",  # sign the grammar rejects
            b"-",  # bare sign
            b"1e",  # digitless exponent
            b"1e+",  # digitless signed exponent
            b'"unterminated',  # unterminated string
            b'"\\q"',  # invalid escape
            b'"\\uZZZZ"',  # invalid \u escape
            b'"a\x01b"',  # raw control character in a string
            b"[1,]",  # trailing array comma
            b"[1 2]",  # missing array delimiter
            b'{"a" 1}',  # missing colon
            b'{"a":1,}',  # trailing object comma
            b"{,}",  # keyless object
            b'{"a":}',  # valueless object
            b"[",  # unclosed array
            b"{",  # unclosed object
            b"]",  # closing bracket with nothing open
            b"}",  # closing brace with nothing open
            b"1 2",  # trailing garbage at the top level
            b"nullx",  # literal followed by junk
            b"tru",  # truncated literal
            b"nan",  # lowercase constants never parse
            b"\xef\xbb\xbf{}",  # a BOM is not whitespace to json.loads
            b"\xff\xfe{}",  # undecodable as UTF-8
            b'{"schema_version": 2}',  # another schema
            b'{"schema_version": "1"}',  # a string is not the constant
            b"[1, 2, 3]",  # not an object
            b"null",  # not an object
            b"5",  # not an object
        ],
        ids=repr,
    )
    def test_rejects(self, raw):
        assert _parse_record(raw) is None

    def test_schema_quirks_compare_like_python(self):
        # 1.0 == 1 and True == 1 through CPython's own comparison.
        assert _parse_record(b'{"schema_version": 1.0, "a": 1}') == {"schema_version": 1.0, "a": 1}
        assert _parse_record(b'{"schema_version": true}') == {"schema_version": True}

    def test_duplicate_keys_keep_first_position_with_last_value(self):
        parsed = _parse_record(b'{"schema_version": 1, "a": 1, "a": 2}')
        assert parsed == {"schema_version": 1, "a": 2}
        assert list(parsed) == ["schema_version", "a"]

    def test_missing_schema_version_is_none(self):
        assert _parse_record(b"{}") is None

    def test_numbers(self):
        parsed = _parse_record(
            b'{"schema_version": 1, "big": 99999999999999999999999999999999,'
            b' "neg": -0, "f": 1e999, "tiny": 1e-400, "e": 1e+5}'
        )
        assert parsed["big"] == 99999999999999999999999999999999
        assert parsed["neg"] == 0
        assert parsed["f"] == float("inf")
        assert parsed["tiny"] == 0.0
        assert parsed["e"] == 100000.0

    def test_constants(self):
        parsed = _parse_record(b'{"schema_version": 1, "n": NaN, "i": Infinity, "m": -Infinity}')
        assert parsed["n"] != parsed["n"]
        assert parsed["i"] == float("inf")
        assert parsed["m"] == float("-inf")

    def test_surrogate_pairing(self):
        # A high surrogate combines only with a low one.
        parsed = _parse_record(
            '{"schema_version": 1, "emoji": "\\ud83d\\ude00", "lone": "\\ud800"}'.encode()
        )
        assert parsed["emoji"] == "\U0001f600"
        assert parsed["lone"] == "\ud800"

    def test_two_high_surrogates_stay_lone(self):
        parsed = _parse_record('{"schema_version": 1, "x": "\\ud800\\ud800"}'.encode())
        assert parsed["x"] == "\ud800\ud800"

    def test_past_budget_nesting_raises_recursion_error(self):
        deep = b"[" * 200_000 + b"]" * 200_000
        with pytest.raises(RecursionError):
            _parse_record(deep)


class TestIdentityFromRecord:
    """The renewal gate — only a verified, well-shaped record renews."""

    def test_none_record_is_none(self):
        assert _identity_from_record(None) is None

    def test_verified_record_answers_identity(self):
        record = {
            "schema_version": 1,
            "verified": True,
            "channel_id": ME_ID,
            "x-goog-authuser": "3",
        }
        identity = _identity_from_record(record)
        assert (identity.slot, identity.channel_id) == (3, ME_ID)

    def test_false_singleton_gates_but_zero_does_not(self):
        good = {"verified": 0, "channel_id": ME_ID, "x-goog-authuser": "1"}
        assert _identity_from_record(good) is not None
        bad = {"verified": False, "channel_id": ME_ID, "x-goog-authuser": "1"}
        assert _identity_from_record(bad) is None

    @pytest.mark.parametrize(
        "channel_id",
        [
            None,
            5,
            "",
            "UC",
            "UCa1b2c3d4e5f6g7h8i9j0k",
            "UCa1b2c3d4e5f6g7h8i9j0k12",
            "XXa1b2c3d4e5f6g7h8i9j0k1",
            "UCa1b2c3d4e5f6g7h8i9j0k1\n",
        ],
    )
    def test_channel_id_must_fullmatch(self, channel_id):
        record = {"verified": True, "channel_id": channel_id, "x-goog-authuser": "1"}
        assert _identity_from_record(record) is None

    @pytest.mark.parametrize("slot", [None, 5, "", "1a", "-1", " 1", "1 "])
    def test_slot_must_be_digits(self, slot):
        record = {"verified": True, "channel_id": ME_ID, "x-goog-authuser": slot}
        assert _identity_from_record(record) is None

    def test_unicode_digit_slot_keeps_its_int_value(self):
        record = {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "٣"}
        identity = _identity_from_record(record)
        assert identity.slot == 3

    def test_isdigit_true_int_false_raises_ints_own_error(self):
        record = {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "\u00b2"}
        with pytest.raises(ValueError, match="invalid literal for int"):
            _identity_from_record(record)


_OWNER_PAYLOAD = b'{\n    "cookie": "SAPISID=a",\n    "x-goog-authuser": "0"\n}'


class TestOwnerOf:
    """The snapshot a renewal must still see at commit time."""

    def test_no_auth_bytes_is_no_owner(self):
        assert _owner_of(None, _record_bytes({"schema_version": 1})) is None

    def test_owner_binds_the_revision_to_these_bytes(self):
        record = {
            "schema_version": 1,
            "auth_sha256": _sha256(_OWNER_PAYLOAD),
            "revision": "ab" * 16,
        }
        owner = _owner_of(_OWNER_PAYLOAD, _record_bytes(record))
        assert (owner.auth_sha256, owner.revision) == (_sha256(_OWNER_PAYLOAD), "ab" * 16)

    def test_foreign_record_answers_null_revision(self):
        record = {"schema_version": 1, "auth_sha256": "00" * 32, "revision": "ab" * 16}
        owner = _owner_of(_OWNER_PAYLOAD, _record_bytes(record))
        assert (owner.auth_sha256, owner.revision) == (_sha256(_OWNER_PAYLOAD), None)

    def test_missing_revision_is_none(self):
        record = {"schema_version": 1, "auth_sha256": _sha256(_OWNER_PAYLOAD)}
        owner = _owner_of(_OWNER_PAYLOAD, _record_bytes(record))
        assert owner.revision is None

    def test_non_string_revision_is_none(self):
        record = {"schema_version": 1, "auth_sha256": _sha256(_OWNER_PAYLOAD), "revision": 7}
        owner = _owner_of(_OWNER_PAYLOAD, _record_bytes(record))
        assert owner.revision is None

    def test_no_record_is_the_null_revision(self):
        owner = _owner_of(_OWNER_PAYLOAD, None)
        assert (owner.auth_sha256, owner.revision) == (_sha256(_OWNER_PAYLOAD), None)


class TestVouchFailure:
    """The decision tree, in the reference's exact order."""

    @staticmethod
    def _read(data: bytes | None, error: Exception | None = None) -> _FileRead:
        return _FileRead(data, (0, 0, 0), error)

    def test_vouched_jar_answers_none(self):
        auth = self._read(b"auth")
        record = self._read(
            _record_bytes(
                {
                    "schema_version": 1,
                    "auth_sha256": _sha256(b"auth"),
                    "stream_sha256": _sha256(b"jar"),
                }
            )
        )
        jar = self._read(b"jar")
        assert _vouch_failure(auth, record, jar) is None

    def test_jar_without_data_is_unreadable_whatever_the_rest(self):
        # The jar gate runs first: even an unreadable record does not win.
        assert (
            _vouch_failure(self._read(b"auth"), self._read(None, OSError("no")), self._read(None))
            == "the stream cookie file is unreadable"
        )

    def test_unreadable_auth(self):
        reason = _vouch_failure(
            self._read(None, OSError("denied")), self._read(None), self._read(b"jar")
        )
        assert reason == "auth.json is unreadable"

    def test_missing_auth(self):
        reason = _vouch_failure(self._read(None), self._read(None), self._read(b"jar"))
        assert reason == "auth.json is missing"

    def test_unreadable_record_renders_its_own_error(self):
        reason = _vouch_failure(
            self._read(b"auth"), self._read(None, OSError("locked")), self._read(b"jar")
        )
        assert reason == "account.json cannot be read (locked)"

    def test_missing_record_is_the_accepted_legacy(self):
        assert _vouch_failure(self._read(b"auth"), self._read(None), self._read(b"jar")) is None

    def test_malformed_record(self):
        reason = _vouch_failure(self._read(b"auth"), self._read(b"not json"), self._read(b"jar"))
        assert reason == "account.json is malformed"

    def test_record_for_other_auth(self):
        record = self._read(
            _record_bytes({"schema_version": 1, "auth_sha256": "00" * 32, "stream_sha256": None})
        )
        reason = _vouch_failure(self._read(b"auth"), record, self._read(b"jar"))
        assert reason == "account.json does not describe the current auth.json"

    def test_record_without_stream_hash_is_the_other_legacy(self):
        record = self._read(_record_bytes({"schema_version": 1, "auth_sha256": _sha256(b"auth")}))
        assert _vouch_failure(self._read(b"auth"), record, self._read(b"jar")) is None

    def test_null_stream_hash_means_no_jar(self):
        record = self._read(
            _record_bytes(
                {"schema_version": 1, "auth_sha256": _sha256(b"auth"), "stream_sha256": None}
            )
        )
        reason = _vouch_failure(self._read(b"auth"), record, self._read(b"jar"))
        assert reason == "the saved sign-in has no stream cookie file"

    def test_mismatched_stream_hash(self):
        record = self._read(
            _record_bytes(
                {
                    "schema_version": 1,
                    "auth_sha256": _sha256(b"auth"),
                    "stream_sha256": _sha256(b"other"),
                }
            )
        )
        reason = _vouch_failure(self._read(b"auth"), record, self._read(b"jar"))
        assert reason == "the stream cookie file does not match the saved sign-in"


def channel_link(browse_id: str, *, page_type: str = "MUSIC_PAGE_TYPE_USER_CHANNEL") -> dict:
    return {
        "compactLinkRenderer": {
            "icon": {"iconType": "ACCOUNT_BOX"},
            "title": {"runs": [{"text": "Your channel"}]},
            "navigationEndpoint": {
                "browseEndpoint": {
                    "browseId": browse_id,
                    "canonicalBaseUrl": "/@wootboop",
                    "browseEndpointContextSupportedConfigs": {
                        "browseEndpointContextMusicConfig": {"pageType": page_type}
                    },
                }
            },
        }
    }


def account_menu(*links: dict, header: bool = True) -> dict:
    items = [
        *links,
        {
            "compactLinkRenderer": {
                "icon": {"iconType": "MONETIZATION_ON"},
                "title": {"runs": [{"text": "Paid memberships"}]},
                "navigationEndpoint": {
                    "browseEndpoint": {"browseId": "FEmemberships_and_purchases"}
                },
            }
        },
        {
            "compactLinkRenderer": {
                "icon": {"iconType": "SWITCH_ACCOUNTS"},
                "title": {"runs": [{"text": "Switch account"}]},
                "navigationEndpoint": None,
            }
        },
        {
            "compactLinkRenderer": {
                "icon": {"iconType": "EXIT_TO_APP"},
                "title": {"runs": [{"text": "Sign out"}]},
                "navigationEndpoint": {"signOutEndpoint": {"hack": True}},
            }
        },
    ]
    return {
        "actions": [
            {
                "openPopupAction": {
                    "popup": {
                        "multiPageMenuRenderer": {
                            "header": {"activeAccountHeaderRenderer": {}} if header else {},
                            "sections": [{"multiPageMenuSectionRenderer": {"items": items}}],
                        }
                    }
                }
            }
        ]
    }


class TestChannelIdFromAccountMenu:
    """The defensive walk — exactly one ACCOUNT_BOX channel link answers."""

    def test_real_shape(self):
        assert _channel_id_from_account_menu(account_menu(channel_link(ME_ID))) == ME_ID

    @pytest.mark.parametrize(
        "response",
        [
            {},
            {"actions": []},
            {"actions": [{}]},
            {"actions": [{"openPopupAction": {}}]},
            {"actions": [{"openPopupAction": {"popup": {}}}]},
            {"actions": [{"openPopupAction": {"popup": {"multiPageMenuRenderer": {}}}}]},
            {"actions": [5]},
            None,
        ],
    )
    def test_malformed_response_is_none(self, response):
        assert _channel_id_from_account_menu(response) is None

    def test_missing_active_account_header_is_none(self):
        assert (
            _channel_id_from_account_menu(account_menu(channel_link(ME_ID), header=False)) is None
        )

    def test_no_channel_link_is_none(self):
        assert _channel_id_from_account_menu(account_menu()) is None

    def test_two_channel_links_are_ambiguous(self):
        menu = account_menu(channel_link(ME_ID), channel_link(OTHER_ID))
        assert _channel_id_from_account_menu(menu) is None

    @pytest.mark.parametrize("page_type", ["MUSIC_PAGE_TYPE_ARTIST", "MUSIC_PAGE_TYPE_ALBUM", None])
    def test_wrong_page_type_is_none(self, page_type):
        assert (
            _channel_id_from_account_menu(account_menu(channel_link(ME_ID, page_type=page_type)))
            is None
        )

    @pytest.mark.parametrize(
        "bad_id", ["", "not-a-channel", "FEmemberships_and_purchases", 5, None]
    )
    def test_non_channel_ids_are_none(self, bad_id):
        link = channel_link(ME_ID)
        link["compactLinkRenderer"]["navigationEndpoint"]["browseEndpoint"]["browseId"] = bad_id
        assert _channel_id_from_account_menu(account_menu(link)) is None

    def test_account_box_without_endpoint_is_none(self):
        link = channel_link(ME_ID)
        link["compactLinkRenderer"]["navigationEndpoint"] = None
        assert _channel_id_from_account_menu(account_menu(link)) is None

    def test_items_must_be_a_list(self):
        menu = account_menu(channel_link(ME_ID))
        menu["actions"][0]["openPopupAction"]["popup"]["multiPageMenuRenderer"]["sections"][0][
            "multiPageMenuSectionRenderer"
        ]["items"] = {"not": "a list"}
        assert _channel_id_from_account_menu(menu) is None

    def test_header_without_the_marker_is_none(self):
        menu = account_menu(channel_link(ME_ID))
        menu["actions"][0]["openPopupAction"]["popup"]["multiPageMenuRenderer"]["header"] = {}
        assert _channel_id_from_account_menu(menu) is None


class TestSerializeHeaders:
    """The one canonical auth.json spelling."""

    def test_matches_the_reference_call(self):
        headers = {"cookie": "SAPISID=a; __Secure-3PAPISID=a", "x-goog-authuser": "0"}
        assert _serialize_headers(headers) == json.dumps(
            headers, ensure_ascii=True, indent=4, sort_keys=True
        ).encode("utf-8")

    def test_keys_sort_regardless_of_insertion_order(self):
        a = _serialize_headers({"b": 1, "a": 2})
        b = _serialize_headers({"a": 2, "b": 1})
        assert a == b

    def test_ensure_ascii_escapes(self):
        payload = _serialize_headers({"k": "é中\U0001f600"})
        assert "é".encode("utf-8") not in payload
        assert b"\\u00e9" in payload and b"\\u4e2d" in payload

    def test_unserialisable_value_raises_the_reference_error(self):
        with pytest.raises(TypeError, match="Object of type set is not JSON serializable"):
            _serialize_headers({"x": {1, 2}})

    def test_unorderable_keys_raise(self):
        with pytest.raises(TypeError, match="'<' not supported"):
            _serialize_headers({1: "a", "b": 2})


class TestCandidatePayload:
    """The candidate auth.json bytes for one browser slot."""

    def test_slot_is_appended_and_serialised(self):
        base = {"cookie": "x", "authorization": "y"}
        payload = _candidate_payload(base, 2)
        assert payload == json.dumps(
            {"cookie": "x", "authorization": "y", "x-goog-authuser": "2"},
            ensure_ascii=True,
            indent=4,
            sort_keys=True,
        ).encode("utf-8")

    def test_existing_slot_value_is_replaced(self):
        base = {"x-goog-authuser": "0", "cookie": "x"}
        payload = _candidate_payload(base, 4)
        merged = json.loads(payload.decode("utf-8"))
        assert merged["x-goog-authuser"] == "4"
        assert merged["cookie"] == "x"
        assert len(merged) == 2


class TestBuildRecord:
    """The account.json assembly."""

    @staticmethod
    def _account(**overrides):
        fields = {"slot": 1, "name": "WootBoop", "handle": "@wootboop", "channel_id": ME_ID}
        fields.update(overrides)
        return _ProbedAccount(**fields)

    def test_assembles_every_field_in_order(self):
        payload = b"payload-bytes"
        record = _build_record(self._account(), payload, "ab" * 32, verified=True)
        assert list(record) == [
            "schema_version",
            "x-goog-authuser",
            "channel_id",
            "name",
            "handle",
            "verified",
            "auth_sha256",
            "stream_sha256",
            "revision",
        ]
        assert record["schema_version"] == 1
        assert record["x-goog-authuser"] == "1"
        assert record["channel_id"] == ME_ID
        assert record["verified"] is True
        assert record["auth_sha256"] == _sha256(payload)
        assert record["stream_sha256"] == "ab" * 32

    def test_falsy_handle_becomes_none(self):
        record = _build_record(self._account(handle=""), b"p", None, verified=True)
        assert record["handle"] is None

    def test_null_channel_id_and_stream_hash_pass_through(self):
        record = _build_record(self._account(channel_id=None), b"p", None, verified=False)
        assert record["channel_id"] is None
        assert record["stream_sha256"] is None
        assert record["verified"] is False

    def test_revisions_are_fresh_and_hexadecimal(self):
        a = _build_record(self._account(), b"p", None, verified=True)["revision"]
        b = _build_record(self._account(), b"p", None, verified=True)["revision"]
        assert a != b
        assert len(a) == 32
        int(a, 16)  # sixteen bytes of hex


class TestSha256Hex:
    """The hash primitive."""

    def test_known_digests(self):
        assert _sha256(b"") == hashlib.sha256(b"").hexdigest()
        assert _sha256(b"abc") == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_str_payload_raises_hashlibs_own_error(self):
        with pytest.raises(TypeError, match="encoded before hashing"):
            _sha256("not bytes")


# ── Native/fallback parity ─────────────────────────────────────────────
# When the Rust extension is built, the native engines must decide
# exactly like the pure-Python references — same accepts, same values,
# same errors with the same messages — for every case, so the two
# implementations cannot drift. The references run in a child
# interpreter with the extension import blocked; every input travels
# as JSON (bytes through their latin-1 spelling, which round-trips
# every byte exactly).

# Each parse case: raw account.json bytes (latin-1 transport), or null.
_PARSE_PARITY_CASES = [
    None,
    b"",
    b"{}",
    b'{"schema_version": 1, "verified": true, "channel_id": "UCa1b2c3d4e5f6g7h8i9j0k1", "x-goog-authuser": "0"}',
    b'{"schema_version": 1.0, "a": [1, 2.5, null, true, NaN, Infinity]}',
    b'{"schema_version": true, "s": "\\ud83d\\ude00", "l": "\\ud800"}',
    b'{"schema_version": 1, "dup": 1, "dup": 2}',
    b'{"schema_version": 1, "big": 99999999999999999999999999999999999}',
    b'{"schema_version": 1, "neg": -0, "f": -0.0, "e": 1E-3, "g": 1e+5, "h": 1e999}',
    b'{"schema_version": 2}',
    b'{"schema_version": "1"}',
    b"[1, 2]",
    b"null",
    b"5",
    b"01",
    b"1.",
    b"+1",
    b"-",
    b"1e",
    b'{"a":}',
    b"[1,]",
    b'{"a" 1}',
    b"[1 2]",
    b"{,}",
    b"[",
    b"{",
    b"]",
    b"1 2",
    b"nullx",
    b"tru",
    b"nan",
    b"\xef\xbb\xbf{}",
    b"\xff\xfe{}",
    b'{"schema_version": 1, "x": "\xc3\xa9"}',
    b'"unterminated',
    b'"\\q"',
    b'"\\uZZZZ"',
    b'"a\x01b"',
    b'{"schema_version": 1, "txt": "\\t\\n\\r\\b\\f\\/\\\\\\""}',
]

# Each identity case: a record dict, or null.
_IDENTITY_PARITY_CASES = [
    None,
    {},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "0"},
    {"verified": False, "channel_id": ME_ID, "x-goog-authuser": "0"},
    {"verified": 0, "channel_id": ME_ID, "x-goog-authuser": "0"},
    {"verified": None, "channel_id": ME_ID, "x-goog-authuser": "0"},
    {"channel_id": ME_ID, "x-goog-authuser": "0"},
    {"verified": True, "x-goog-authuser": "0"},
    {"verified": True, "channel_id": ME_ID},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "12"},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "\u0663"},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": "1a"},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": ""},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": " 1"},
    {"verified": True, "channel_id": ME_ID, "x-goog-authuser": 1},
    {"verified": True, "channel_id": "UCa1b2c3d4e5f6g7h8i9j0k12", "x-goog-authuser": "1"},
    {"verified": True, "channel_id": "XXa1b2c3d4e5f6g7h8i9j0k1", "x-goog-authuser": "1"},
    {"verified": True, "channel_id": 5, "x-goog-authuser": "1"},
    {"verified": True, "channel_id": None, "x-goog-authuser": "1"},
]

# Each owner case: [auth bytes (latin-1), record bytes (latin-1)], either null.
_OWNER_PARITY_CASES = [
    [None, None],
    [None, b'{"schema_version": 1}'],
    [b"auth", None],
    [b"auth", b"not json"],
    [
        b"auth",
        b'{"schema_version": 1, "auth_sha256": "'
        + hashlib.sha256(b"auth").hexdigest().encode()
        + b'", "revision": "ab"}',
    ],
    [
        b"auth",
        b'{"schema_version": 1, "auth_sha256": "'
        + hashlib.sha256(b"auth").hexdigest().encode()
        + b'"}',
    ],
    [
        b"auth",
        b'{"schema_version": 1, "auth_sha256": "'
        + hashlib.sha256(b"auth").hexdigest().encode()
        + b'", "revision": 7}',
    ],
    [b"auth", b'{"schema_version": 1, "auth_sha256": "00"}'],
    [
        b'{"k": "\\u00e9"}',
        b'{"schema_version": 1, "auth_sha256": "'
        + hashlib.sha256('{"k": "é"}'.encode()).hexdigest().encode()
        + b'", "revision": "xy"}',
    ],
]

# Each vouch case: [auth spec, record spec, jar spec] where a spec is
# {"data": latin-1-or-null, "err": message-or-null}.
_VOUCH_PARITY_CASES = [
    [{"data": "auth", "err": None}, {"data": None, "err": None}, {"data": "jar", "err": None}],
    [{"data": "auth", "err": None}, {"data": None, "err": "locked"}, {"data": "jar", "err": None}],
    [
        {"data": "auth", "err": None},
        {"data": "not json", "err": None},
        {"data": "jar", "err": None},
    ],
    [{"data": "auth", "err": None}, {"data": None, "err": None}, {"data": None, "err": "gone"}],
    [{"data": None, "err": "denied"}, {"data": None, "err": None}, {"data": "jar", "err": None}],
    [{"data": None, "err": None}, {"data": None, "err": None}, {"data": "jar", "err": None}],
]


def _vouch_record_for(auth: bytes) -> bytes:
    return (
        b'{"schema_version": 1, "auth_sha256": "'
        + hashlib.sha256(auth).hexdigest().encode()
        + b'", "stream_sha256": "'
        + hashlib.sha256(b"jar").hexdigest().encode()
        + b'"}'
    )


_VOUCH_PARITY_CASES += [
    [
        {"data": "auth", "err": None},
        {"data": _vouch_record_for(b"auth"), "err": None},
        {"data": "jar", "err": None},
    ],
    [
        {"data": "auth", "err": None},
        {"data": b'{"schema_version": 1}', "err": None},
        {"data": "jar", "err": None},
    ],
    [
        {"data": "auth", "err": None},
        {
            "data": b'{"schema_version": 1, "auth_sha256": "'
            + hashlib.sha256(b"auth").hexdigest().encode()
            + b'"}',
            "err": None,
        },
        {"data": "jar", "err": None},
    ],
    [
        {"data": "auth", "err": None},
        {
            "data": b'{"schema_version": 1, "auth_sha256": "'
            + hashlib.sha256(b"auth").hexdigest().encode()
            + b'", "stream_sha256": null}',
            "err": None,
        },
        {"data": "jar", "err": None},
    ],
    [
        {"data": "auth", "err": None},
        {
            "data": b'{"schema_version": 1, "auth_sha256": "'
            + hashlib.sha256(b"auth").hexdigest().encode()
            + b'", "stream_sha256": "00"}',
            "err": None,
        },
        {"data": "jar", "err": None},
    ],
]

# Each menu case: a full response dict.
_MENU_PARITY_CASES = [
    account_menu(channel_link(ME_ID)),
    account_menu(),
    account_menu(channel_link(ME_ID), channel_link(OTHER_ID)),
    account_menu(channel_link(ME_ID, page_type="MUSIC_PAGE_TYPE_ARTIST")),
    account_menu(channel_link("not-a-channel")),
    account_menu(channel_link(ME_ID), header=False),
    {},
    {"actions": []},
    {"actions": [{}]},
    {
        "actions": [
            {
                "openPopupAction": {
                    "popup": {
                        "multiPageMenuRenderer": {
                            "header": {"activeAccountHeaderRenderer": {}},
                            "sections": [{"multiPageMenuSectionRenderer": {"items": "not a list"}}],
                        }
                    }
                }
            }
        ]
    },
    {
        "actions": [
            {
                "openPopupAction": {
                    "popup": {
                        "multiPageMenuRenderer": {
                            "header": [1, 2],
                            "sections": [
                                {"multiPageMenuSectionRenderer": {"items": [channel_link(ME_ID)]}}
                            ],
                        }
                    }
                }
            }
        ]
    },
    {
        "actions": [
            {
                "openPopupAction": {
                    "popup": {
                        "multiPageMenuRenderer": {
                            "header": {"other": 1},
                            "sections": [
                                {"multiPageMenuSectionRenderer": {"items": [channel_link(ME_ID)]}}
                            ],
                        }
                    }
                }
            }
        ]
    },
]

# Each serialize case: a headers dict of JSON-round-trippable values.
_SERIALIZE_PARITY_CASES = [
    {},
    {"cookie": "SAPISID=a; __Secure-3PAPISID=a", "x-goog-authuser": "0"},
    {"b": 1, "a": "x", "c": None, "d": True, "e": 1.5, "f": [1, {"z": 2}], "g": (3, 4)},
    {"n": "é中\U0001f600", "esc": 'quote"back\\slash\ttab'},
    {"n": 1e999, "z": -0.0, "big": 10**30},
    {"nested": {"deep": {"deeper": [1, [2, [3]]]}}},
]

# Each candidate case: [base headers, slot].
_CANDIDATE_PARITY_CASES = [
    [{"cookie": "x", "authorization": "y"}, 0],
    [{"cookie": "x", "authorization": "y"}, 2],
    [{"x-goog-authuser": "0", "cookie": "x"}, 4],
    [{}, 0],
    [{"k": "v"}, 12],
]

# Each build case: [slot, name, handle, channel_id, verified, payload, stream_sha].
_BUILD_PARITY_CASES = [
    [1, "WootBoop", "@wootboop", ME_ID, True, b"payload", "ab" * 32],
    [0, "Someone", "", None, False, b"{}", None],
    [4, "Ünïcode", "@hand", ME_ID, True, b"bytes-\xff", "cd" * 32],
]


def _canonical(value):
    """A JSON-safe canonical form of a decoded value — NaN-tagged floats,
    type-tagged scalars, sorted dict items — lists only, so the child's
    JSON wire format compares equal to the in-process form."""
    if isinstance(value, dict):
        return ["d", sorted([_canonical(k), _canonical(v)] for k, v in value.items())]
    if isinstance(value, (list, tuple)):
        return ["l", [_canonical(v) for v in value]]
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, int):
        return ["i", str(value)]
    if isinstance(value, float):
        if value != value:
            return ["f", "nan"]
        if value == float("inf"):
            return ["f", "inf"]
        if value == float("-inf"):
            return ["f", "-inf"]
        return ["f", repr(value)]
    if value is None:
        return ["n", None]
    return ["s", repr(value)]


# The child interpreter runs this program with the case matrix on stdin
# and prints the reference engines' outcomes as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.auth import (
    _FileRead,
    _build_record,
    _candidate_payload,
    _channel_id_from_account_menu,
    _identity_from_record,
    _owner_of,
    _parse_record,
    _ProbedAccount,
    _serialize_headers,
    _sha256,
    _vouch_failure,
)


def to_bytes(latin1):
    return None if latin1 is None else latin1.encode("latin-1")


def outcome(fn):
    def run(*args):
        try:
            return {"ok": fn(*args)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return run


def canon(value):
    if isinstance(value, dict):
        return ["d", sorted((canon(k), canon(v)) for k, v in value.items())]
    if isinstance(value, (list, tuple)):
        return ["l", [canon(v) for v in value]]
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, int):
        return ["i", str(value)]
    if isinstance(value, float):
        if value != value:
            return ["f", "nan"]
        if value == float("inf"):
            return ["f", "inf"]
        if value == float("-inf"):
            return ["f", "-inf"]
        return ["f", repr(value)]
    if value is None:
        return ["n", None]
    return ["s", repr(value)]


def identity_outcome(record):
    ident = _identity_from_record(record)
    return None if ident is None else [ident.slot, ident.channel_id]


def owner_outcome(auth, record_raw):
    owner = _owner_of(to_bytes(auth), to_bytes(record_raw))
    return None if owner is None else [owner.auth_sha256, owner.revision]


def vouch_outcome(auth, record, jar):
    def read(spec):
        err = OSError(spec["err"]) if spec["err"] is not None else None
        return _FileRead(to_bytes(spec["data"]), (0, 0, 0), err)
    return _vouch_failure(read(auth), read(record), read(jar))


def build_outcome(slot, name, handle, channel_id, verified, payload, stream_sha256):
    account = _ProbedAccount(slot=slot, name=name, handle=handle, channel_id=channel_id)
    record = _build_record(account, to_bytes(payload), stream_sha256, verified=verified)
    record.pop("revision")  # fresh randomness, compared separately
    return record


payload = json.load(sys.stdin)
out = {
    "parse": [outcome(lambda raw: canon(_parse_record(to_bytes(raw))))(c) for c in payload["parse"]],
    "identity": [outcome(identity_outcome)(c) for c in payload["identity"]],
    "owner": [outcome(owner_outcome)(*c) for c in payload["owner"]],
    "vouch": [outcome(vouch_outcome)(*c) for c in payload["vouch"]],
    "menu": [outcome(_channel_id_from_account_menu)(c) for c in payload["menu"]],
    "serialize": [outcome(lambda h: _serialize_headers(h).hex())(c) for c in payload["serialize"]],
    "candidate": [outcome(lambda h, s: _candidate_payload(h, s).hex())(*c) for c in payload["candidate"]],
    "build": [outcome(build_outcome)(*c) for c in payload["build"]],
    "sha": [outcome(lambda b: _sha256(to_bytes(b)))(c) for c in payload["sha"]],
}
json.dump(out, sys.stdout)
"""


def _latin1(raw: bytes | None) -> str | None:
    return None if raw is None else raw.decode("latin-1")


def _parity_matrix() -> dict:
    return {
        "parse": [_latin1(c) for c in _PARSE_PARITY_CASES],
        "identity": _IDENTITY_PARITY_CASES,
        "owner": [[_latin1(a), _latin1(r)] for a, r in _OWNER_PARITY_CASES],
        "vouch": [
            [{"data": _latin1(to_bytes_or_none(spec)), "err": spec_err(spec)} for spec in case]
            for case in _VOUCH_PARITY_CASES
        ],
        "menu": _MENU_PARITY_CASES,
        "serialize": _SERIALIZE_PARITY_CASES,
        "candidate": _CANDIDATE_PARITY_CASES,
        "build": [
            [slot, name, handle, channel_id, verified, _latin1(payload), stream_sha]
            for slot, name, handle, channel_id, verified, payload, stream_sha in _BUILD_PARITY_CASES
        ],
        "sha": [_latin1(b""), _latin1(b"abc"), _latin1("é".encode("utf-8")), _latin1(b"x" * 1000)],
    }


def to_bytes_or_none(spec) -> bytes | None:
    data = spec.get("data") if isinstance(spec, dict) else spec
    if data is None:
        return None
    return data if isinstance(data, bytes) else data.encode("latin-1")


def spec_err(spec) -> str | None:
    return spec.get("err") if isinstance(spec, dict) else None


def _run_outcome(fn):
    def run(*args):
        try:
            return {"ok": fn(*args)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    return run


def _native_outcomes() -> dict:
    n = _ytm_player_native

    def identity(record):
        ident = n.identity_from_record(record)
        return None if ident is None else [ident[0], ident[1]]

    def owner(auth, record_raw):
        owner = n.owner_of(to_bytes_or_none(auth), to_bytes_or_none(record_raw))
        return None if owner is None else [owner[0], owner[1]]

    def vouch(auth, record, jar):
        def data(spec):
            return to_bytes_or_none(spec)

        def err(spec):
            e = spec_err(spec)
            return None if e is None else OSError(e)

        return n.vouch_failure(
            data(auth), err(auth), data(record), err(record), data(jar), err(jar)
        )

    def build(slot, name, handle, channel_id, verified, payload, stream_sha256):
        record = dict(
            n.build_record(
                slot, name, handle, channel_id, verified, to_bytes_or_none(payload), stream_sha256
            )
        )
        record.pop("revision")  # fresh randomness, compared separately
        return record

    return {
        "parse": [
            _run_outcome(lambda raw: _canonical(n.parse_record(to_bytes_or_none(raw))))(c)
            for c in [_latin1(x) for x in _PARSE_PARITY_CASES]
        ],
        "identity": [_run_outcome(identity)(c) for c in _IDENTITY_PARITY_CASES],
        "owner": [_run_outcome(owner)(*[_latin1(x) for x in case]) for case in _OWNER_PARITY_CASES],
        "vouch": [_run_outcome(vouch)(*case) for case in _VOUCH_PARITY_CASES],
        "menu": [_run_outcome(n.channel_id_from_account_menu)(c) for c in _MENU_PARITY_CASES],
        "serialize": [
            _run_outcome(lambda h: n.serialize_headers(h).hex())(c) for c in _SERIALIZE_PARITY_CASES
        ],
        "candidate": [
            _run_outcome(lambda h, s: n.candidate_payload(h, s).hex())(*c)
            for c in _CANDIDATE_PARITY_CASES
        ],
        "build": [
            _run_outcome(build)(*c)
            for c in [
                [slot, name, handle, channel_id, verified, _latin1(payload), stream_sha]
                for slot, name, handle, channel_id, verified, payload, stream_sha in _BUILD_PARITY_CASES
            ]
        ],
        "sha": [
            _run_outcome(lambda b: n.sha256_hex(to_bytes_or_none(b)))(c)
            for c in [
                _latin1(b""),
                _latin1(b"abc"),
                _latin1("é".encode("utf-8")),
                _latin1(b"x" * 1000),
            ]
        ],
    }


class TestNativeParity:
    """When the Rust extension is built, the native engines must decide
    exactly like the pure-Python references — same accepts, same values,
    same errors with the same messages — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        payload = json.dumps(_parity_matrix())
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        cases = _parity_matrix()
        for engine in native:
            for case, n_out, f_out in zip(cases[engine], native[engine], fallback[engine]):
                assert n_out == f_out, (
                    f"native and fallback {engine} outcomes differ for {case!r}:\n"
                    f"native:   {n_out!r}\n"
                    f"fallback: {f_out!r}"
                )
