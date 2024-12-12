# %%
from dotenv import load_dotenv
import os
from datetime import datetime
import requests

load_dotenv()
slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

headers = {
    "Content-type": "application/json",
}

json_data = {
    "text": "Hello, World!",
}

response = requests.post(
    f"https://hooks.slack.com/services/T083HUW4LQL/B083J1T3404/{slack_webhook_url}",
    headers=headers,
    json=json_data,
)


print(response.content)
# %%
