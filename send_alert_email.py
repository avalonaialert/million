#!/usr/bin/env python3
"""Send the alert email written by generate_supertrend.py --alert-file through Gmail.

Usage: send_alert_email.py ALERT_FILE
  ALERT_FILE       plain-text alert; its first line is the subject. ALERT_FILE with a
                   .html suffix, if present, is sent as the HTML version.
Environment:
  GMAIL_USERNAME      Gmail address that sends the email
  GMAIL_APP_PASSWORD  app password for that account (not the normal password)
  ALERT_EMAIL_TO      recipient(s), comma-separated (default: GMAIL_USERNAME)
"""

from __future__ import annotations

import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        print("No new signals; no email sent.")
        return 0

    user = os.environ.get("GMAIL_USERNAME", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not user or not password:
        print("GMAIL_USERNAME and GMAIL_APP_PASSWORD must be set (as repository secrets).")
        return 1
    recipients = [a.strip() for a in os.environ.get("ALERT_EMAIL_TO", "").split(",") if a.strip()] or [user]

    subject, _, body = path.read_text(encoding="utf-8").partition("\n")
    message = EmailMessage()
    message["Subject"] = subject.strip()
    message["From"] = user
    message["To"] = ", ".join(recipients)
    message.set_content(body.lstrip("\n"))
    html_path = path.with_suffix(".html")
    if html_path.exists():
        message.add_alternative(html_path.read_text(encoding="utf-8"), subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)
    print(f"Sent alert email to {len(recipients)} recipient(s): {message['Subject']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
