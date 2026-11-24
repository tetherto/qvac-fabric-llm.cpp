#!/usr/bin/env python3
import importlib.util
import io
import pathlib
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


send_report = load_module("send_report", "send-report.py")


class SendReportTest(unittest.TestCase):
    def test_configuration_detection(self):
        self.assertFalse(send_report.configured({}))
        self.assertFalse(send_report.configured({"MAIL_TO": "a@x", "SMTP_HOST": "h"}))
        self.assertTrue(send_report.configured({"MAIL_TO": "a@x, b@x", "SMTP_HOST": "h", "MAIL_FROM": "f@x"}))
        self.assertEqual(send_report.recipients({"MAIL_TO": "a@x, b@x,"}), ["a@x", "b@x"])

    def test_message_has_body_and_attachments(self):
        with tempfile.TemporaryDirectory() as tmp:
            attachment = pathlib.Path(tmp) / "summary.json"
            attachment.write_text("{}")
            message = send_report.build_message("subject", "body text", [str(attachment)], "f@x", ["a@x", "b@x"])
        self.assertEqual(message["Subject"], "subject")
        self.assertEqual(message["To"], "a@x, b@x")
        parts = list(message.iter_attachments())
        self.assertEqual([part.get_filename() for part in parts], ["summary.json"])
        self.assertEqual(parts[0].get_content_type(), "application/json")
        self.assertIn("body text", message.get_body().get_content())

    def test_missing_attachment_is_skipped(self):
        message = send_report.build_message("s", "b", ["/nonexistent/summary.json"], "f@x", ["a@x"])
        self.assertEqual(list(message.iter_attachments()), [])

    def test_port_selects_implicit_tls_or_starttls(self):
        message = send_report.build_message("s", "b", [], "f@x", ["a@x"])
        with mock.patch.object(send_report, "send_implicit_tls") as implicit, mock.patch.object(send_report, "send_starttls") as starttls:
            send_report.send(message, {"SMTP_HOST": "h", "SMTP_PORT": "465", "SMTP_USER": "u", "SMTP_PASSWORD": "p"})
            implicit.assert_called_once_with(message, "h", 465, "u", "p")
            send_report.send(message, {"SMTP_HOST": "h", "SMTP_PORT": "587"})
            starttls.assert_called_once_with(message, "h", 587, "", "")

    def test_tls_verifies_certificates_and_hostnames(self):
        context = send_report.tls_context()
        self.assertEqual(context.verify_mode, send_report.ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        message = send_report.build_message("s", "b", [], "f@x", ["a@x"])
        with mock.patch.object(send_report.smtplib, "SMTP_SSL") as smtp_ssl:
            send_report.send_implicit_tls(message, "h", 465, "", "")
            self.assertTrue(smtp_ssl.call_args.kwargs["context"].check_hostname)
        with mock.patch.object(send_report.smtplib, "SMTP") as smtp:
            send_report.send_starttls(message, "h", 587, "", "")
            session = smtp.return_value.__enter__.return_value
            self.assertTrue(session.starttls.call_args.kwargs["context"].check_hostname)

    def test_main_skips_when_unconfigured_and_dry_runs_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = pathlib.Path(tmp) / "body.md"
            body.write_text("hello")
            with mock.patch.object(send_report, "send") as send:
                send_report.main(["--subject", "s", "--body-file", str(body)], env={})
                send.assert_not_called()
                out = io.StringIO()
                with mock.patch("sys.stdout", out):
                    send_report.main(["--subject", "s", "--body-file", str(body), "--dry-run"],
                                     env={"MAIL_TO": "a@x", "SMTP_HOST": "h", "MAIL_FROM": "f@x"})
                send.assert_not_called()
                self.assertIn("Subject: s", out.getvalue())
                self.assertIn("hello", out.getvalue())


if __name__ == "__main__":
    unittest.main()
