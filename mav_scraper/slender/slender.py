# %%
from dotenv import load_dotenv
import os
from slack_sdk import WebClient
from loguru import logger

from mav_scraper.slender import slender

load_dotenv()


class _Slender:
    def __init__(self):
        self.client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
        logger.add("console.log", rotation="1 MB", level="INFO")
        logger.add("error.log", rotation="1 MB", level="ERROR")

    def get_channel_id(self, channel_name):
        for channel in self.client.conversations_list()["channels"]:
            if channel["name"] == channel_name:
                return channel["id"]
        return None

    def send_message(self, message: str, channel_name="mav-scraping-monitoring"):
        self.client.chat_postMessage(
            channel=self.get_channel_id(channel_name=channel_name),
            text=message,
        )

    def success(self, success_message: str):
        self.send_message(
            f"✅ {success_message}", channel_name="mav-scraping-monitoring"
        )
        logger.success(success_message)

    def info(self, info_message: str):
        # self.send_message(f"ℹ️ {info_message}", channel_name="mav-scraping-monitoring")
        logger.info(info_message)

    def warning(self, warning_message: str):
        # self.send_message(
        # f"⚠️ {warning_message}", channel_name="mav-scraping-monitoring"
        # )
        # self.send_message(f"⚠️ {warning_message}", channel_name="mav-warning")
        logger.warning(warning_message)

    def error(self, error_message: str):
        # self.send_message(f"❌ {error_message}", channel_name="mav-scraping-monitoring")
        # self.send_message(f"❌ {error_message}", channel_name="mav-warning")
        self.send_message(f"❌ {error_message}", channel_name="mav-error")
        logger.error(error_message)


# %%
if __name__ == "__main__":
    slender.send_message("Hello, world!")
# %%
