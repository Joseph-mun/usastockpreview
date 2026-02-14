# -*- coding: utf-8 -*-
"""Telegram Bot API client for sending prediction results."""

import argparse
import time

import requests

from src.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, get_logger

logger = get_logger(__name__)

API_BASE = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    """Send messages and images via Telegram Bot API."""

    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = API_BASE.format(token=self.token)

    def send_text(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message. Returns True on success."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json().get("ok", False)
        except requests.RequestException as e:
            logger.error("Telegram sendMessage failed: %s", e)
            return False

    def send_photo(self, image_bytes: bytes, caption: str = None) -> bool:
        """Send a photo (PNG bytes). Returns True on success."""
        url = f"{self.base_url}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes, "image/png")}
        data = {"chat_id": self.chat_id}
        if caption:
            data["caption"] = caption[:1024]  # Telegram caption limit
            data["parse_mode"] = "HTML"
        try:
            resp = requests.post(url, data=data, files=files, timeout=60)
            resp.raise_for_status()
            return resp.json().get("ok", False)
        except requests.RequestException as e:
            logger.error("Telegram sendPhoto failed: %s", e)
            return False

    def send_prediction_report(
        self,
        summary_text: str,
        chart_bytes: bytes = None,
        table_text: str = None,
    ):
        """
        Send full prediction report: summary → chart → table.
        Sends messages with 1-second gaps to respect rate limits.
        """
        self.send_text(summary_text)

        if chart_bytes:
            time.sleep(1)
            self.send_photo(chart_bytes, caption="Probability Trend (60 days)")

        if table_text:
            time.sleep(1)
            self.send_text(table_text)


def main():
    """CLI entry point for sending ad-hoc notifications."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--notify-training", type=str, help="Training job status")
    args = parser.parse_args()

    if args.notify_training:
        notifier = TelegramNotifier()
        status = args.notify_training
        emoji = "\u2705" if status == "success" else "\u274c"
        notifier.send_text(f"{emoji} Weekly Training: {status}")


if __name__ == "__main__":
    main()
