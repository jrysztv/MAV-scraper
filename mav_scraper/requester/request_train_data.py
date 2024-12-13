# %%
from __future__ import annotations
import asyncio
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from dotenv import load_dotenv
from mav_scraper.slender import slender as logger

load_dotenv()


class TrainDataFetcher:
    """
    Handles fetching train location and schedule data from MÁV (Hungarian State Railways) endpoints.
    """

    BASE_URL: str = "https://vonatinfo.mav-start.hu/map.aspx/getData"

    def __init__(self, headers: Dict[str, str], concurrency=3) -> None:
        """
        Initialize the TrainDataFetcher with required headers and cookies.

        :param headers: HTTP headers to use for requests.
        :param cookies: HTTP cookies to use for requests.
        """
        self.headers = headers
        # self.cookies = cookies
        self.semaphore = asyncio.Semaphore(concurrency)

    def fetch_train_locations(self) -> pd.DataFrame:
        """
        Fetches the current train locations from MÁV.

        :return: A DataFrame containing train locations with columns:
                timestamp, delay, lat, relation, train_number,
                service_type, line, lon, elvira_id.
                Returns an empty DataFrame in case of errors.
        """
        json_data = {"a": "TRAINS", "jo": {"history": False, "id": False}}
        response = requests.post(self.BASE_URL, headers=self.headers, json=json_data)

        if response.status_code != 200:
            logger.error(f"Failed to fetch train locations: {response.status_code}")
            return pd.DataFrame()

        result = response.json().get("d", {}).get("result", {})
        data = result.get("Trains", {}).get("Train", [])
        creation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not data or not creation_time:
            logger.error("Invalid data structure returned for train locations.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.columns = [col.replace("@", "") for col in df.columns]
        df.insert(
            0, "Timestamp", pd.to_datetime(creation_time).strftime("%Y-%m-%d %H:%M:%S")
        )

        df.rename(
            columns={
                "Timestamp": "timestamp",
                "Delay": "delay",
                "Lat": "lat",
                "Relation": "relation",
                "TrainNumber": "train_number",
                "Menetvonal": "service_type",
                "Line": "line",
                "Lon": "lon",
                "ElviraID": "elvira_id",
            },
            inplace=True,
        )

        # Check if all elvira_id are NaN
        if df["elvira_id"].isna().all():
            logger.error("All Elvira IDs are NaN. Aborting process.")
            raise ValueError("All Elvira IDs are NaN.")

        # Log error for train_numbers with NaN elvira_id
        nan_elvira_ids = df[df["elvira_id"].isna()]["train_number"].tolist()
        if nan_elvira_ids:
            logger.error(f"NaN ElviraID for the following train(s): {nan_elvira_ids}")

        # Filter out rows without elvira_id
        df = df[df["elvira_id"].notna()]

        if df.empty:
            logger.error("No valid ElviraID after filtering.")
            raise ValueError("No valid ElviraID.")

        final_columns = [
            "timestamp",
            "train_number",
            "elvira_id",
            "delay",
            "lat",
            "lon",
            "relation",
            "service_type",
        ]

        df = df[final_columns]
        return df.query('service_type != "HEV"')

    def fetch_train_details(
        self, elvira_id: str, train_number: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches HTML details for a specific train from MÁV.

        :param elvira_id: The Elvira ID of the train.
        :param train_number: The train number.
        :param async_fetch: If True, use asynchronous HTTP fetch. Otherwise, use synchronous requests.
        :return: A dictionary containing 'elvira_id', 'train_number', and 'html',
                 or None on error.
        """
        json_data = {
            "a": "TRAIN",
            "jo": {
                "vsz": train_number,
                "zoom": False,
                "csakkozlekedovonat": True,
            },
        }

        # Synchronous logic (as before)
        response = requests.post(self.BASE_URL, headers=self.headers, json=json_data)
        if response.status_code != 200:
            logger.error(
                f"Failed to fetch train details for {train_number}: {response.status_code}"
            )
            return None

        html = response.json()["d"]["result"]["html"]
        return {"elvira_id": elvira_id, "train_number": train_number, "html": html}

    async def _fetch_train_details_async(
        self,
        client: httpx.AsyncClient,
        elvira_id: str,
        train_number: str,
    ) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            json_data = {
                "a": "TRAIN",
                "jo": {
                    "vsz": train_number,
                    "zoom": False,
                    "csakkozlekedovonat": True,
                },
            }

            for i in range(3):
                try:
                    if i > 0:
                        logger.warning(
                            f"Retrying train details fetch for {train_number} ({i})"
                        )
                    response = await client.post(self.BASE_URL, json=json_data)
                    break
                except Exception as e:
                    logger.error(f"Error fetching details for {train_number}: {e}")
                    await asyncio.sleep(1)
            else:
                return None

            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch train details async for {train_number}: {response.status_code}"
                )
                return None

            data = response.json()
            html = data["d"]["result"]["html"]
            return {"elvira_id": elvira_id, "train_number": train_number, "html": html}


class ScheduleParser:
    """
    Handles parsing of HTML schedule data fetched from MÁV and converting it into structured DataFrames.
    """

    @staticmethod
    def parse_schedule_table(
        elvira_id: str, train_number: str, html: str
    ) -> Dict[str, Union[str, pd.DataFrame]]:
        """
        Parses the schedule table HTML for a given train.

        :param elvira_id: The Elvira ID of the train.
        :param train_number: The train number.
        :param html: HTML content containing the train schedule table.
        :return: A dictionary with 'elvira_id', 'train_number', and 'data' (DataFrame).
                 The DataFrame columns:
                 [elvira_id, train_number, last_request, Km, Station, Scheduled Arrival,
                  Estimated Arrival, Scheduled Departure, Estimated Departure, Platform, Status]
                 Returns an empty DataFrame if no table is found or on error.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", class_="vt")
            if not table:
                logger.warning(f"Schedule table not found for train {train_number}")
                return {
                    "elvira_id": elvira_id,
                    "train_number": train_number,
                    "data": pd.DataFrame(),
                }

            rows = table.find_all("tr")[4:]  # Skip header rows
            data = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 5:
                    km = cols[0].text.strip()
                    station = cols[1].text.strip()
                    scheduled_arrival, estimated_arrival = (
                        ScheduleParser._extract_schedule_time(cols[2])
                    )
                    scheduled_departure, estimated_departure = (
                        ScheduleParser._extract_schedule_time(cols[3])
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

            df.insert(0, "elvira_id", elvira_id)
            df.insert(1, "train_number", train_number)
            df.insert(
                2, "last_request", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            return {"elvira_id": elvira_id, "train_number": train_number, "data": df}

        except Exception as e:
            logger.error(f"Error parsing schedule for train {train_number}: {e}")
            return {
                "elvira_id": elvira_id,
                "train_number": train_number,
                "data": pd.DataFrame(),
            }

    @staticmethod
    def _extract_schedule_time(cell) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts scheduled and estimated times from a table cell.

        :param cell: A BeautifulSoup Tag representing the cell.
        :return: A tuple (scheduled_time, estimated_time)
        """
        scheduled = cell.contents[0].strip() if cell.contents else None
        estimated_tag = cell.find("span", style="color:red")
        estimated = estimated_tag.text.strip() if estimated_tag else scheduled
        return scheduled, estimated


class DataStorageManager:
    """
    Handles storage of train schedules and locations in Parquet or CSV formats.
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialize the DataStorageManager with a base directory for output.

        :param base_dir: The base directory for storing data.
        """
        self.base_dir = base_dir
        self.data_dir = self.base_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Directories for various outputs
        self.parquet_store_dir = self.data_dir / "parquet_store"
        self.parquet_store_dir.mkdir(parents=True, exist_ok=True)

        self.parsed_train_schedule_dir = self.data_dir / "parsed_train_schedules"
        self.parsed_train_schedule_dir.mkdir(parents=True, exist_ok=True)

        self.spot_train_locations_dir = self.data_dir / "spot_train_locations"
        self.spot_train_locations_dir.mkdir(parents=True, exist_ok=True)

        # Parquet file paths
        self.stored_parsed_train_schedules_path = (
            self.parquet_store_dir / "parsed_train_schedules.parquet"
        )
        self.stored_spot_train_locations_path = (
            self.parquet_store_dir / "spot_train_locations.parquet"
        )

    def store_as_parquet(
        self,
        parsed_train_schedules: List[Dict[str, Any]],
        spot_train_locations: pd.DataFrame,
    ) -> None:
        """
        Saves parsed train schedules and spot train locations to Parquet files, updating
        existing storage if they exist.

        :param parsed_train_schedules: List of dictionaries containing parsed train schedule data.
        :param spot_train_locations: DataFrame containing current spot train locations.
        """
        if not parsed_train_schedules:
            logger.warning("No parsed train schedules to store.")
            return

        parsed_train_schedules_df = pd.concat(
            [entry["data"] for entry in parsed_train_schedules], ignore_index=True
        )

        # Ensure initial parquet files are created if not present
        self._initialize_parquet_files(spot_train_locations, parsed_train_schedules_df)

        parsed_train_schedules_df = self._update_parsed_train_schedule_storage(
            parsed_train_schedules_df
        )

        # Update spot train locations
        stored_spot_train_locations_df = self._update_spot_train_locations_storage(
            spot_train_locations
        )

        # Save updated data back to parquet
        parsed_train_schedules_df.to_parquet(
            self.stored_parsed_train_schedules_path, compression="snappy"
        )
        stored_spot_train_locations_df.to_parquet(
            self.stored_spot_train_locations_path, compression="snappy"
        )

        # Compute unique train count
        unique_train_identifier_count = len(
            parsed_train_schedules_df[["elvira_id", "train_number"]].drop_duplicates()
        )
        logger.info(
            f"{unique_train_identifier_count} train schedules saved to {self.parsed_train_schedule_dir}"
        )
        logger.info(f"Spot train locations saved to {self.spot_train_locations_dir}")

    def store_as_csv(
        self,
        parsed_train_schedules: List[Dict[str, Any]],
        spot_train_locations: pd.DataFrame,
    ) -> None:
        """
        Saves parsed train schedules and spot train locations to CSV files.

        :param parsed_train_schedules: List of dictionaries containing parsed train schedule data.
        :param spot_train_locations: DataFrame containing current spot train locations.
        """
        if not parsed_train_schedules:
            logger.warning("No parsed train schedules to store.")
            return

        # Save schedules
        for schedule_entry in parsed_train_schedules:
            schedule_entry_file_path = (
                self.parsed_train_schedule_dir
                / f"{schedule_entry['train_number']}_{schedule_entry['elvira_id']}.csv"
            )
            if schedule_entry_file_path.exists():
                logger.warning(
                    f"File {schedule_entry_file_path} already exists and will be overwritten."
                )
            schedule_entry["data"].to_csv(schedule_entry_file_path, index=False)

        # Save spot train locations
        spot_filename = (
            self.spot_train_locations_dir
            / f"spot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if spot_filename.exists():
            logger.warning(
                f"File {spot_filename} already exists and will be overwritten."
            )
        spot_train_locations.to_csv(spot_filename, index=False)

        unique_trains = {
            (p["elvira_id"], p["train_number"]) for p in parsed_train_schedules
        }
        logger.info(f"{len(unique_trains)} train schedules saved as CSV.")
        logger.info("Spot train locations saved as CSV.")

    def _update_spot_train_locations_storage(
        self, spot_train_locations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the stored spot train locations by appending current data.

        :param spot_train_locations: DataFrame with current spot train locations.
        :return: Updated DataFrame with appended spot train locations.
        """
        stored_spot_train_locations = pd.read_parquet(
            self.stored_spot_train_locations_path
        )
        return pd.concat(
            [stored_spot_train_locations, spot_train_locations], ignore_index=True
        )

    def _update_parsed_train_schedule_storage(
        self, current_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the stored parsed train schedules with the current data, ensuring that for each
        (elvira_id, train_number) pair, only the newest table is kept.

        :param stored_df: DataFrame with previously stored schedules.
        :param current_df: DataFrame with currently parsed schedules for one or more trains.
        :return: Updated DataFrame that replaces any existing schedules for each (elvira_id, train_number)
                found in current_df with the newest version from current_df.
        """
        key_columns = ["elvira_id", "train_number"]

        # Update parsed train schedules
        stored_df = pd.read_parquet(self.stored_parsed_train_schedules_path)

        # If there are no schedules to process, simply return stored_df
        if current_df.empty:
            logger.info("No new schedule data provided. Stored data remains unchanged.")
            return stored_df

        # Identify unique (elvira_id, train_number) pairs in the current data
        current_pairs = current_df[key_columns].drop_duplicates()

        # For each pair, we will remove any old entries from stored_df and then append the new data
        updated_stored_df = stored_df.copy()
        replaced_count = 0
        added_count = 0

        for _, row in current_pairs.iterrows():
            elvira_id = row["elvira_id"]
            train_number = row["train_number"]

            # Check if this (elvira_id, train_number) already exists in stored_df
            mask = (updated_stored_df["elvira_id"] == elvira_id) & (
                updated_stored_df["train_number"] == train_number
            )
            if mask.any():
                # Remove old version
                updated_stored_df = updated_stored_df[~mask]
                replaced_count += 1
                logger.info(
                    f"Replacing previously stored schedule for train_number={train_number}, elvira_id={elvira_id} with the newer version."
                )
            else:
                added_count += 1
                logger.info(
                    f"Adding new schedule for train_number={train_number}, elvira_id={elvira_id}."
                )

        # Append the current schedules
        updated_stored_df = pd.concat(
            [updated_stored_df, current_df], ignore_index=True
        )

        logger.info(
            f"Schedule update complete: {added_count} new schedule(s) added, {replaced_count} existing schedule(s) replaced."
        )

        return updated_stored_df

    def _initialize_parquet_files(
        self, spot_train_locations: pd.DataFrame, current_df: pd.DataFrame
    ) -> None:
        """
        Ensures the initial parquet storage files exist. If not, creates them.

        :param spot_train_locations: DataFrame containing current spot train locations.
        :param current_df: DataFrame containing current parsed train schedules.
        """
        if not self.stored_parsed_train_schedules_path.exists():
            logger.warning(
                f"{self.stored_parsed_train_schedules_path} does not exist. Creating..."
            )
            current_df.to_parquet(self.stored_parsed_train_schedules_path)
        if not self.stored_spot_train_locations_path.exists():
            logger.warning(
                f"{self.stored_spot_train_locations_path} does not exist. Creating..."
            )
            spot_train_locations.to_parquet(self.stored_spot_train_locations_path)


class HungarianRailwayScraperPipeline:
    """
    Orchestrates the entire workflow of:
    1. Fetching train locations.
    2. Fetching train details and parsing schedules.
    3. Storing data to either Parquet or CSV.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """
        Initialize the HungarianRailwayPipeline.

        :param base_dir: Optional base directory for data storage.
                         Defaults to two levels up from this file.
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir

        # Configure logging
        # logger.add("console.log", rotation="1 MB", retention="1 day", level="INFO")
        # logger.add("error.log", rotation="10 MB", retention="30 days", level="ERROR")

        # HTTP headers and cookies
        headers = {
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

        self.train_data_fetcher = TrainDataFetcher(headers)
        self.schedule_parser = ScheduleParser()
        self.storage_handler = DataStorageManager(self.base_dir)

    def fetch_and_parse_all_trains(
        self, limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Fetch and parse train data:
        1. Fetch train locations (returns a DataFrame).
        2. Fetch train details for each train and parse the schedules.

        :param limit: Number of trains to limit processing to, for testing or performance reasons.
        :return: A tuple containing:
                 - DataFrame of spot train locations
                 - List of parsed train schedule dictionaries
        """
        train_locations_df = self.train_data_fetcher.fetch_train_locations()
        if train_locations_df.empty:
            logger.error("No train locations fetched.")
            return pd.DataFrame(), []

        train_list = train_locations_df[["elvira_id", "train_number"]].to_dict(
            orient="records"
        )
        if limit is not None:
            train_list = train_list[:limit]

        # Fetch details
        train_details = []
        for train in tqdm(train_list, desc="Fetching train details"):
            details = self.train_data_fetcher.fetch_train_details(**train)
            if details:
                train_details.append(details)

        # Parse schedules
        parsed_train_schedules = []
        for details in tqdm(train_details, desc="Parsing train schedules"):
            parsed = self.schedule_parser.parse_schedule_table(**details)
            if not parsed["data"].empty:
                parsed_train_schedules.append(parsed)
            else:
                logger.warning(
                    f"Parsed schedule for train {details['train_number']} is empty."
                )

        return train_locations_df, parsed_train_schedules

    async def fetch_and_parse_all_trains_async(
        self, limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        train_locations_df = self.train_data_fetcher.fetch_train_locations()
        if train_locations_df.empty:
            logger.error("No train locations fetched.")
            return pd.DataFrame(), []

        train_list = train_locations_df[["elvira_id", "train_number"]].to_dict(
            orient="records"
        )
        if limit is not None:
            train_list = train_list[:limit]

        async with httpx.AsyncClient(headers=self.train_data_fetcher.headers) as client:
            tasks = [
                self.train_data_fetcher._fetch_train_details_async(
                    client, t["elvira_id"], t["train_number"]
                )
                for t in train_list
            ]

            train_details = await tqdm_asyncio.gather(*tasks)

        train_details = [d for d in train_details if d is not None]

        parsed_train_schedules = []
        for details in train_details:
            parsed = self.schedule_parser.parse_schedule_table(**details)
            if not parsed["data"].empty:
                parsed_train_schedules.append(parsed)
            else:
                logger.warning(
                    f"Parsed schedule for train {details['train_number']} is empty."
                )

        return train_locations_df, parsed_train_schedules

    def run(
        self,
        limit: Optional[int] = None,
        save_format: str = "parquet",
        async_mode: bool = False,
    ) -> None:
        """
        Main workflow for scraping MÁV train data:
        1. Fetch and parse train data.
        2. Save results in the specified format (Parquet or CSV).

        :param limit: Optional limit on the number of trains to process.
        :param save_format: The format to save the results in, either "parquet" or "csv".
        :param async_mode: If True, run the pipeline asynchronously. If False, run it synchronously.
        """
        try:
            logger.info("Starting Hungarian Railway Data Pipeline")

            if async_mode:
                # Run the async pipeline
                train_locations_df, parsed_train_schedules = asyncio.run(
                    self.fetch_and_parse_all_trains_async(limit)
                )
            else:
                # Run the synchronous pipeline
                train_locations_df, parsed_train_schedules = (
                    self.fetch_and_parse_all_trains(limit)
                )

            if train_locations_df.empty:
                logger.error("No data to process. Exiting.")
                return

            save_format = save_format.lower()
            if save_format == "parquet":
                self.storage_handler.store_as_parquet(
                    parsed_train_schedules, train_locations_df
                )
            elif save_format == "csv":
                self.storage_handler.store_as_csv(
                    parsed_train_schedules, train_locations_df
                )
            else:
                logger.error(
                    f"Unsupported save format: {save_format}. Use 'parquet' or 'csv'."
                )

            logger.success("Data pipeline completed successfully")
        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}")
            raise e


# %% Run the pipeline
if __name__ == "__main__":
    pipeline = HungarianRailwayScraperPipeline()
    pipeline.run(
        limit=None, save_format="parquet", async_mode=True
    )  # Set `limit=None` to process all trains

# %%
