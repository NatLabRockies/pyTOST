from __future__ import annotations

import io
import json

import pytest

from pyTOST.mcp import content
from pyTOST.mcp.server import _read_message, _write_message, dispatch


class TestDispatch:
    def test_initialize_reports_protocol_version(self, tmp_path):
        result = dispatch("initialize", None, tmp_path)
        assert result["protocolVersion"]
        assert result["serverInfo"]["name"] == "pytost-mcp"

    def test_tools_list_includes_expected_tools(self, tmp_path):
        result = dispatch("tools/list", None, tmp_path)
        names = {tool["name"] for tool in result["tools"]}
        assert names == {"overview", "engine_guide", "strengths_and_weaknesses", "read_doc"}

    def test_tools_call_overview_returns_text(self, tmp_path):
        result = dispatch("tools/call", {"name": "overview"}, tmp_path)
        text = result["content"][0]["text"]
        assert "pyTOST" in text

    def test_tools_call_engine_guide_returns_text(self, tmp_path):
        result = dispatch("tools/call", {"name": "engine_guide"}, tmp_path)
        text = result["content"][0]["text"]
        assert "cluster" in text

    def test_tools_call_strengths_and_weaknesses_returns_text(self, tmp_path):
        result = dispatch("tools/call", {"name": "strengths_and_weaknesses"}, tmp_path)
        text = result["content"][0]["text"]
        assert "Strengths" in text

    def test_tools_call_read_doc_reads_file(self, tmp_path):
        (tmp_path / "README.md").write_text("hello readme", encoding="utf-8")
        result = dispatch(
            "tools/call", {"name": "read_doc", "arguments": {"name": "readme"}}, tmp_path
        )
        assert result["content"][0]["text"] == "hello readme"

    def test_tools_call_read_doc_missing_file(self, tmp_path):
        result = dispatch(
            "tools/call", {"name": "read_doc", "arguments": {"name": "readme"}}, tmp_path
        )
        assert "Missing file" in result["content"][0]["text"]

    def test_tools_call_unknown_tool_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("tools/call", {"name": "nope"}, tmp_path)

    def test_unsupported_method_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported method"):
            dispatch("bogus/method", None, tmp_path)

    def test_doc_resources_map_is_non_empty(self):
        assert content.DOC_RESOURCES


class TestFraming:
    """Regression guard: MCP stdio uses newline-delimited JSON, not
    LSP-style Content-Length header framing."""

    def test_write_message_uses_newline_delimited_json_not_headers(self):
        stream = io.BytesIO()
        _write_message({"jsonrpc": "2.0", "id": 1, "result": {}}, stdout=stream)
        raw = stream.getvalue()
        assert b"Content-Length" not in raw
        assert raw.endswith(b"\n")
        assert json.loads(raw.decode("utf-8").strip()) == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {},
        }

    def test_read_message_parses_single_ndjson_line(self):
        payload = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "initialize"}) + "\n"
        stream = io.BytesIO(payload.encode("utf-8"))
        message = _read_message(stdin=stream)
        assert message == {"jsonrpc": "2.0", "id": 2, "method": "initialize"}

    def test_read_message_returns_none_at_eof(self):
        stream = io.BytesIO(b"")
        assert _read_message(stdin=stream) is None

    def test_read_message_skips_blank_lines(self):
        stream = io.BytesIO(b"\n")
        assert _read_message(stdin=stream) == {}
