#!/usr/bin/env python3
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import narrate


class NarrateTest(unittest.TestCase):
    def test_request_shape(self):
        payload = narrate.build_request("report body", "claude-opus-5-5")
        self.assertEqual(payload["model"], "claude-opus-5-5")
        self.assertEqual(payload["max_tokens"], narrate.MAX_TOKENS)
        self.assertEqual(payload["fallbacks"], "default")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertIn("report body", payload["messages"][0]["content"])
        self.assertEqual(narrate.request_headers("k")["x-api-key"], "k")
        self.assertEqual(narrate.request_headers("k")["anthropic-beta"], narrate.FALLBACK_BETA)

    def test_text_blocks_are_joined_and_refusals_are_empty(self):
        response = {"stop_reason": "end_turn", "content": [{"type": "thinking", "thinking": ""},
                                                            {"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]}
        self.assertEqual(narrate.narrative_text(response), "line one\nline two")
        self.assertEqual(narrate.narrative_text({"stop_reason": "refusal", "content": [{"type": "text", "text": "no"}]}), "")

    def test_narrate_uses_the_transport(self):
        seen = {}

        def transport(payload, api_key):
            seen["payload"], seen["key"] = payload, api_key
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": "summary"}]}

        self.assertEqual(narrate.narrate("report", "model-x", "key-y", transport), "summary")
        self.assertEqual(seen["key"], "key-y")
        self.assertEqual(seen["payload"]["model"], "model-x")

    def test_main_without_key_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "report.md"
            report.write_text("x")
            out = pathlib.Path(tmp) / "narrative.md"
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
                narrate.main(["--report", str(report), "--out", str(out)])
            self.assertFalse(out.exists())

    def test_main_writes_narrative_and_survives_api_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = pathlib.Path(tmp) / "report.md"
            report.write_text("x")
            out = pathlib.Path(tmp) / "narrative.md"
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "k"}):
                with mock.patch.object(narrate, "call_api", return_value={"stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}):
                    narrate.main(["--report", str(report), "--out", str(out)])
                self.assertEqual(out.read_text(), "ok\n")
                out.unlink()
                with mock.patch.object(narrate, "call_api", side_effect=OSError("down")):
                    narrate.main(["--report", str(report), "--out", str(out)])
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
