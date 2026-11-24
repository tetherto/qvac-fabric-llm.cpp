#!/usr/bin/env python3
"""E-mail a benchmark report through an SMTP relay.

usage: send-report.py --subject TEXT --body-file FILE [--attach FILE]... [--dry-run]

Configuration comes from the environment (config.env, loaded by weekly.sh): MAIL_TO (comma-separated), MAIL_FROM,
SMTP_HOST, SMTP_PORT (465 for implicit TLS, otherwise STARTTLS), SMTP_USER and SMTP_PASSWORD. Without MAIL_TO or
SMTP_HOST the message is not sent and the exit code is 0, so an unconfigured machine still completes its run.
"""
from __future__ import annotations

import argparse
import email.message
import os
import pathlib
import smtplib
import ssl
import sys

IMPLICIT_TLS_PORT = 465
DEFAULT_SMTP_PORT = 587
SMTP_TIMEOUT_SECONDS = 120
ATTACHMENT_TYPES = {".json": ("application", "json"), ".md": ("text", "markdown"), ".txt": ("text", "plain")}
DEFAULT_ATTACHMENT_TYPE = ("application", "octet-stream")


def recipients(env):
    return [address.strip() for address in env.get("MAIL_TO", "").split(",") if address.strip()]


def configured(env):
    return bool(recipients(env) and env.get("SMTP_HOST") and env.get("MAIL_FROM"))


def attachment_type(path):
    return ATTACHMENT_TYPES.get(path.suffix, DEFAULT_ATTACHMENT_TYPE)


def attach(message, path):
    if not path.is_file():
        print(f"send-report.py: attachment {path} is missing, sending without it", file=sys.stderr)
        return
    maintype, subtype = attachment_type(path)
    message.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)


def build_message(subject, body, attachments, sender, to):
    message = email.message.EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(to)
    message.set_content(body)
    for path in attachments:
        attach(message, pathlib.Path(path))
    return message


def login(session, user, password):
    if user:
        session.login(user, password)


def tls_context():
    return ssl.create_default_context()


def send_implicit_tls(message, host, port, user, password):
    with smtplib.SMTP_SSL(host, port, context=tls_context(), timeout=SMTP_TIMEOUT_SECONDS) as session:
        login(session, user, password)
        session.send_message(message)


def send_starttls(message, host, port, user, password):
    with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT_SECONDS) as session:
        session.starttls(context=tls_context())
        login(session, user, password)
        session.send_message(message)


def send(message, env):
    host, port = env["SMTP_HOST"], int(env.get("SMTP_PORT") or DEFAULT_SMTP_PORT)
    user, password = env.get("SMTP_USER", ""), env.get("SMTP_PASSWORD", "")
    if port == IMPLICIT_TLS_PORT:
        send_implicit_tls(message, host, port, user, password)
    else:
        send_starttls(message, host, port, user, password)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--attach", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None, env=None):
    args = parse_args(argv)
    env = os.environ if env is None else env
    if not configured(env):
        print("send-report.py: MAIL_TO, MAIL_FROM or SMTP_HOST not set, not sending", file=sys.stderr)
        return
    body = pathlib.Path(args.body_file).read_text()
    message = build_message(args.subject, body, args.attach, env["MAIL_FROM"], recipients(env))
    if args.dry_run:
        sys.stdout.write(message.as_string())
        return
    send(message, env)
    print(f"sent '{args.subject}' to {', '.join(recipients(env))}")


if __name__ == "__main__":
    main()
