# %%
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import datetime


class MAVScraper:
    BASE_URL = "https://vonatinfo.mav-start.hu/map.aspx/getData"

    def __init__(self):
        self.headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Content-Type": "application/json; charset=UTF-8",
            "DNT": "1",
            "Origin": "https://vonatinfo.mav-start.hu",
            "Referer": "https://vonatinfo.mav-start.hu/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "X-Requested-With": "XMLHttpRequest",
            "sec-ch-ua": '"Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }
        self.cookies = {"ASP.NET_SessionId": "pwz3swnbq5ghpwjiforyi52n"}
        self.base_dir = Path(__file__).resolve().parent.parent
        # Console log: short retention, smaller size
        logger.add(
            "console.log",
            rotation="1 MB",  # Rotate when file size exceeds 1 MB
            retention="1 day",  # Keep logs for 1 day
            level="INFO",  # Log INFO and above
        )

        # Error log: longer retention, larger size
        logger.add(
            "error.log",
            rotation="10 MB",  # Rotate when file size exceeds 10 MB
            retention="30 days",  # Keep logs for 30 days
            level="ERROR",  # Log ERROR and above
        )

    def fetch_train_locations(self):
        json_data = {
            "a": "TRAINS",
            "jo": {"history": False, "id": False},
        }

        response = requests.post(
            self.BASE_URL, cookies=self.cookies, headers=self.headers, json=json_data
        )
        if response.status_code != 200:
            logger.error(f"Failed to fetch train locations: {response.status_code}")
            return pd.DataFrame()

        data = response.json()["d"]["result"]["Trains"]["Train"]
        creation_time = response.json()["d"]["result"]["@CreationTime"]
        df = pd.DataFrame(data)
        df.columns = [col.replace("@", "") for col in df.columns]
        df.insert(0, "Timestamp", pd.to_datetime(creation_time))
        df.columns = [
            "timestamp",
            "delay",
            "lat",
            "relation",
            "train_number",
            "service_type",
            "line",
            "lon",
            "elvira_id",
        ]
        return df

    def fetch_train_details(self, elvira_id, train_number):
        json_data = {
            "a": "TRAIN",
            "jo": {
                "v": elvira_id,
                "vsz": train_number,
                "zoom": False,
                "csakkozlekedovonat": True,
            },
        }

        response = requests.post(self.BASE_URL, headers=self.headers, json=json_data)
        if response.status_code != 200:
            logger.error(
                f"Failed to fetch train details for {train_number}: {response.status_code}"
            )
            return None

        html = response.json()["d"]["result"]["html"]
        return {"elvira_id": elvira_id, "train_number": train_number, "html": html}

    def parse_schedule_table(self, elvira_id, train_number, html):
        try:
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", class_="vt")
            if not table:
                logger.error(f"Schedule table not found for train {train_number}")
                return pd.DataFrame()

            rows = table.find_all("tr")[4:]  # Skip header rows
            data = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    km = cols[0].text.strip()
                    station = cols[1].text.strip()
                    scheduled_arrival, estimated_arrival = self._extract_time(cols[2])
                    scheduled_departure, estimated_departure = self._extract_time(
                        cols[3]
                    )
                    platform = cols[4].text.strip()
                    status = (
                        "passed" if "row_past" in row.get("class", []) else "not passed"
                    )
                    data.append(
                        [
                            km,
                            station,
                            scheduled_arrival,
                            estimated_arrival,
                            scheduled_departure,
                            estimated_departure,
                            platform,
                            status,
                        ]
                    )

            df = pd.DataFrame(
                data,
                columns=[
                    "Km",
                    "Station",
                    "Scheduled Arrival",
                    "Estimated Arrival",
                    "Scheduled Departure",
                    "Estimated Departure",
                    "Platform",
                    "Status",
                ],
            )
            return {"elvira_id": elvira_id, "train_number": train_number, "data": df}

        except Exception as e:
            logger.error(f"Error parsing schedule for train {train_number}: {e}")
            return {
                "elvira_id": elvira_id,
                "train_number": train_number,
                "data": pd.DataFrame(),
            }

    def save_to_csv(self, parsed_train_schedules, spot_train_locations):
        parsed_train_schedule_dir = self.base_dir / "data" / "parsed_train_schedules"
        parsed_train_schedule_dir.mkdir(parents=True, exist_ok=True)

        spot_train_locations_dir = self.base_dir / "data" / "spot_train_locations"
        spot_train_locations_dir.mkdir(parents=True, exist_ok=True)

        for schedule_entry in parsed_train_schedules:
            schedule_entry_file_path = (
                parsed_train_schedule_dir
                / f"{schedule_entry['train_number']}_{schedule_entry['elvira_id']}.csv"
            )
            if schedule_entry_file_path.exists():
                logger.warning(
                    f"File {schedule_entry_file_path} already exists and will be overwritten."
                )
            schedule_entry["data"].to_csv(schedule_entry_file_path, index=False)

        spot_train_locations_filename = (
            spot_train_locations_dir
            / f"spot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if spot_train_locations_filename.exists():
            logger.warning(
                f"File {spot_train_locations_filename} already exists and will be overwritten."
            )

        spot_train_locations.to_csv(spot_train_locations_filename, index=False)

    @staticmethod
    def _extract_time(cell):
        scheduled = cell.contents[0].strip() if cell.contents else None
        estimated = (
            cell.find("span", style="color:red").text.strip()
            if cell.find("span", style="color:red")
            else scheduled
        )
        return scheduled, estimated

    def run(self, limit=None):
        """
        Main workflow for scraping MÁV train data.
        :param limit: Limit the number of trains to process (for testing).
        """
        logger.info("Starting MAV Scraper")

        # Step 1: Fetch train locations
        train_locations_data_frame = self.fetch_train_locations()
        if train_locations_data_frame.empty:
            logger.error("No train locations fetched. Exiting.")
            return

        # Limit the number of trains if a limit is specified
        train_list = train_locations_data_frame[["elvira_id", "train_number"]].to_dict(
            orient="records"
        )
        if limit is not None:
            train_list = train_list[:limit]

        # Step 2: Fetch train details
        train_details = []
        for train in tqdm(train_list, desc="Fetching train details"):
            details = self.fetch_train_details(**train)
            if details:
                train_details.append(details)

        # Step 3: Parse train schedules
        parsed_train_schedules = []
        for details in tqdm(train_details, desc="Parsing train schedules"):
            try:
                parsed = self.parse_schedule_table(**details)
                if not parsed["data"].empty:  # Check if the DataFrame is not empty
                    parsed_train_schedules.append(parsed)
                else:
                    logger.warning(
                        f"Parsed schedule for train {details['train_number']} is empty."
                    )
            except Exception as e:
                logger.error(
                    f"An error occurred while parsing schedule for train {details['train_number']}: {e}"
                )

        # Step 4: Save results to CSV
        self.save_to_csv(parsed_train_schedules, train_locations_data_frame)

        logger.info("Scraper completed successfully")


# %%
if __name__ == "__main__":
    scraper = MAVScraper()

    # Run the scraper workflow
    scraper.run(limit=None)  # Set `limit=None` to process all trains

# %%
