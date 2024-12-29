# %%
from __future__ import annotations
import asyncio
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
import polyline
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
    Also handles fetching line (shape) data in one go using the 'LINE' endpoint.
    """

    BASE_URL: str = "https://vonatinfo.mav-start.hu/map.aspx/getData"

    def __init__(self, headers: Dict[str, str], concurrency=3) -> None:
        """
        Initialize the TrainDataFetcher with required headers and concurrency limit (for async).
        """
        self.headers = headers
        self.semaphore = asyncio.Semaphore(concurrency)

    def fetch_train_locations(self) -> pd.DataFrame:
        """
        Fetches the current train locations from MÁV.

        :return: DataFrame with columns [timestamp, train_number, elvira_id, delay, lat, lon, relation, service_type].
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
        # Exclude "HEV" trains if you don't need them
        return df.query('service_type != "HEV"')

    def fetch_train_details(
        self, elvira_id: str, train_number: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches HTML details for a specific train from MÁV.
        """
        json_data = {
            "a": "TRAIN",
            "jo": {
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

    async def _fetch_train_details_async(
        self,
        client: httpx.AsyncClient,
        elvira_id: str,
        train_number: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Async version of fetch_train_details.
        """
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

    def fetch_all_lines(self) -> pd.DataFrame:
        """
        Fetch all lines at once from MÁV 'LINE' endpoint and decode their shapes.
        Returns a DataFrame with [line_id, lat, lon].
        """
        # Modify these bounds/padding as desired (from your second code)
        padding = 10
        json_data = {
            "a": "LINE",
            "jo": {
                "sw": [45.7 - padding, 16.1 - padding],
                "ne": [48.6 + padding, 22.9 + padding],
                "id": "bg",
                "hidden": True,
                "history": True,
                "zoom": 1,
            },
        }

        try:
            resp = requests.post(self.BASE_URL, headers=self.headers, json=json_data)
            resp.raise_for_status()
            lines = resp.json()["d"]["result"]["lines"]

            # Convert each line's polyline into a DataFrame
            frames = []
            for line in lines:
                shape_points = polyline.decode(line["points"])
                # Build a DataFrame for this line
                df_line = pd.DataFrame(shape_points, columns=["lat", "lon"])
                df_line.insert(0, "line_id", line["linenum"])
                frames.append(df_line)

            if frames:
                return pd.concat(frames, ignore_index=True)
            else:
                logger.warning("No lines retrieved.")
                return pd.DataFrame()

        except Exception as exc:
            logger.error(f"Failed to fetch lines: {exc}")
            return pd.DataFrame()


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
        """
        scheduled = cell.contents[0].strip() if cell.contents else None
        estimated_tag = cell.find("span", style="color:red")
        estimated = estimated_tag.text.strip() if estimated_tag else scheduled
        return scheduled, estimated


class DataStorageManager:
    """
    Handles storage of train schedules, shapes, and locations in Parquet or CSV formats.
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialize the DataStorageManager with a base directory for output.
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

        # Updated for storing shapes by line_id instead of train:
        self.parsed_line_shapes_dir = self.data_dir / "parsed_line_shapes"
        self.parsed_line_shapes_dir.mkdir(parents=True, exist_ok=True)

        # Parquet file paths
        self.stored_parsed_train_schedules_path = (
            self.parquet_store_dir / "parsed_train_schedules.parquet"
        )
        self.stored_spot_train_locations_path = (
            self.parquet_store_dir / "spot_train_locations.parquet"
        )
        # Updated: store lines by line_id
        self.stored_parsed_line_shapes_path = (
            self.parquet_store_dir / "parsed_line_shapes.parquet"
        )

    def store_as_parquet(
        self,
        parsed_train_schedules: List[Dict[str, Any]],
        line_shapes: pd.DataFrame,
        spot_train_locations: pd.DataFrame,
    ) -> None:
        """
        Saves parsed train schedules, line shapes, and spot train locations to Parquet files, updating
        existing storage if they exist.
        """
        # If there are schedule data
        if parsed_train_schedules:
            schedules_df = pd.concat(
                [entry["data"] for entry in parsed_train_schedules], ignore_index=True
            )
        else:
            schedules_df = pd.DataFrame()

        # Ensure initial parquet files exist
        self._initialize_parquet_files(spot_train_locations, schedules_df, line_shapes)

        # Update train schedules
        if not schedules_df.empty:
            schedules_df = self._update_parsed_train_schedule_storage(schedules_df)
            schedules_df.to_parquet(
                self.stored_parsed_train_schedules_path, compression="snappy"
            )

            unique_train_identifier_count = len(
                schedules_df[["elvira_id", "train_number"]].drop_duplicates()
            )
            logger.info(
                f"{unique_train_identifier_count} train schedules saved/updated."
            )
        else:
            logger.warning("No parsed train schedules to store.")

        # Update line shapes
        if not line_shapes.empty:
            line_shapes_updated = self._update_parsed_line_shape_storage(line_shapes)
            line_shapes_updated.to_parquet(
                self.stored_parsed_line_shapes_path, compression="snappy"
            )
            logger.info("Line shapes saved/updated.")
        else:
            logger.warning("No line shapes data to store.")

        # Update spot train locations
        stored_spot_train_locations = self._update_spot_train_locations_storage(
            spot_train_locations
        )
        stored_spot_train_locations.to_parquet(
            self.stored_spot_train_locations_path, compression="snappy"
        )
        logger.info("Spot train locations saved/updated.")

    def store_as_csv(
        self,
        parsed_train_schedules: List[Dict[str, Any]],
        line_shapes: pd.DataFrame,
        spot_train_locations: pd.DataFrame,
    ) -> None:
        """
        Saves parsed train schedules, line shapes, and spot train locations to CSV files.
        """
        # Schedules
        if parsed_train_schedules:
            for schedule_entry in parsed_train_schedules:
                schedule_file = (
                    self.parsed_train_schedule_dir
                    / f"{schedule_entry['train_number']}_{schedule_entry['elvira_id']}.csv"
                )
                schedule_entry["data"].to_csv(schedule_file, index=False)
            unique_trains = {
                (p["elvira_id"], p["train_number"]) for p in parsed_train_schedules
            }
            logger.info(f"{len(unique_trains)} train schedules saved as CSV.")
        else:
            logger.warning("No parsed train schedules to store in CSV.")

        # Line shapes
        if not line_shapes.empty:
            # We'll store each line_id in a separate CSV or just store one big file
            # One big file example:
            lines_filename = self.parsed_line_shapes_dir / "lines_all.csv"
            line_shapes.to_csv(lines_filename, index=False)
            logger.info("Line shapes saved as CSV.")
        else:
            logger.warning("No line shapes data to store in CSV.")

        # Spot train locations
        if not spot_train_locations.empty:
            spot_filename = (
                self.spot_train_locations_dir
                / f"spot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            spot_train_locations.to_csv(spot_filename, index=False)
            logger.info("Spot train locations saved as CSV.")
        else:
            logger.warning("No spot train locations to store in CSV.")

    def _initialize_parquet_files(
        self,
        spot_train_locations: pd.DataFrame,
        schedules_df: pd.DataFrame,
        line_shapes: pd.DataFrame,
    ) -> None:
        """
        Ensures the initial parquet storage files exist. If not, creates them.
        """
        if not self.stored_parsed_train_schedules_path.exists():
            logger.warning(
                f"{self.stored_parsed_train_schedules_path} does not exist. Creating..."
            )
            if not schedules_df.empty:
                schedules_df.to_parquet(self.stored_parsed_train_schedules_path)

        if not self.stored_parsed_line_shapes_path.exists():
            logger.warning(
                f"{self.stored_parsed_line_shapes_path} does not exist. Creating..."
            )
            if not line_shapes.empty:
                line_shapes.to_parquet(self.stored_parsed_line_shapes_path)

        if not self.stored_spot_train_locations_path.exists():
            logger.warning(
                f"{self.stored_spot_train_locations_path} does not exist. Creating..."
            )
            if not spot_train_locations.empty:
                spot_train_locations.to_parquet(self.stored_spot_train_locations_path)

    def _update_parsed_train_schedule_storage(
        self, current_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the stored parsed train schedules with the current data, ensuring that for each
        (elvira_id, train_number) pair, only the newest schedule is kept.
        """
        key_columns = ["elvira_id", "train_number"]
        try:
            stored_df = pd.read_parquet(self.stored_parsed_train_schedules_path)
        except Exception:
            logger.warning("No existing train schedules parquet found. Creating new.")
            return current_df

        if current_df.empty:
            logger.info("No new schedule data provided.")
            return stored_df

        current_pairs = current_df[key_columns].drop_duplicates()

        updated_stored_df = stored_df.copy()
        replaced_count = 0
        added_count = 0

        for _, row in current_pairs.iterrows():
            elvira_id = row["elvira_id"]
            train_number = row["train_number"]

            # Remove old version for that pair if it exists
            mask = (updated_stored_df["elvira_id"] == elvira_id) & (
                updated_stored_df["train_number"] == train_number
            )
            if mask.any():
                updated_stored_df = updated_stored_df[~mask]
                replaced_count += 1
            else:
                added_count += 1

        # Append the current schedules
        updated_stored_df = pd.concat(
            [updated_stored_df, current_df], ignore_index=True
        )

        logger.info(f"Schedule update: {added_count} new, {replaced_count} replaced.")
        return updated_stored_df

    def _update_parsed_line_shape_storage(
        self, current_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates the stored line shape data by `line_id`, overwriting any previously stored shape for that line.
        """
        key_col = "line_id"
        try:
            stored_df = pd.read_parquet(self.stored_parsed_line_shapes_path)
        except Exception:
            logger.warning("No existing line shapes parquet found. Creating new.")
            return current_df

        if current_df.empty:
            logger.info("No new line shape data provided.")
            return stored_df

        current_line_ids = current_df[key_col].drop_duplicates()

        updated_stored_df = stored_df.copy()
        replaced_count = 0
        added_count = 0

        for line_id in current_line_ids:
            mask = updated_stored_df[key_col] == line_id
            if mask.any():
                updated_stored_df = updated_stored_df[~mask]
                replaced_count += 1
            else:
                added_count += 1

        updated_stored_df = pd.concat(
            [updated_stored_df, current_df], ignore_index=True
        )
        logger.info(
            f"Line shape update: {added_count} new line(s), {replaced_count} replaced."
        )
        return updated_stored_df

    def _update_spot_train_locations_storage(
        self, spot_train_locations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Appends the new spot train locations to the stored DataFrame.
        """
        try:
            stored_spot_train_locations = pd.read_parquet(
                self.stored_spot_train_locations_path
            )
        except Exception:
            logger.warning(
                "No existing spot train locations parquet found. Creating new."
            )
            return spot_train_locations

        # Simple append
        return pd.concat(
            [stored_spot_train_locations, spot_train_locations], ignore_index=True
        )


class HungarianRailwayScraperPipeline:
    """
    Orchestrates the entire workflow of:
    1. Fetching train locations.
    2. Fetching train details and parsing schedules.
    3. Fetching *all* line shapes with a single 'LINE' request and processing them.
    4. Storing data to either Parquet or CSV.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """
        Initialize the HungarianRailwayPipeline.
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir

        # HTTP headers
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
        Fetch train locations, fetch train details for each train, parse schedules.
        (Synchronous version, ignoring shapes by train, since we do single-call shapes below.)
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

        # Fetch details and parse schedules
        train_details = []
        for train in tqdm(train_list, desc="Fetching train details"):
            details = self.train_data_fetcher.fetch_train_details(**train)
            if details:
                train_details.append(details)

        parsed_schedules = []
        for details in tqdm(train_details, desc="Parsing schedules"):
            parsed = self.schedule_parser.parse_schedule_table(**details)
            if not parsed["data"].empty:
                parsed_schedules.append(parsed)
            else:
                logger.warning(
                    f"Parsed schedule for train {details['train_number']} is empty."
                )

        return train_locations_df, parsed_schedules

    async def fetch_and_parse_all_trains_async(
        self, limit: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Async version for fetching/parsing train schedules.
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

        async with httpx.AsyncClient(headers=self.train_data_fetcher.headers) as client:
            tasks = [
                self.train_data_fetcher._fetch_train_details_async(
                    client, t["elvira_id"], t["train_number"]
                )
                for t in train_list
            ]
            train_details = await tqdm_asyncio.gather(*tasks)

        train_details = [d for d in train_details if d is not None]

        parsed_schedules = []
        for details in train_details:
            parsed = self.schedule_parser.parse_schedule_table(**details)
            if not parsed["data"].empty:
                parsed_schedules.append(parsed)
            else:
                logger.warning(
                    f"Parsed schedule for train {details['train_number']} is empty."
                )

        return train_locations_df, parsed_schedules

    def run(
        self,
        limit: Optional[int] = None,
        save_format: str = "parquet",
        async_mode: bool = False,
    ) -> None:
        """
        Main workflow for scraping MÁV train data:
        1. Fetch & parse train schedules (either sync or async).
        2. Fetch line shapes in one request, parse them.
        3. Save results in 'parquet' or 'csv'.
        """
        try:
            logger.info("Starting Hungarian Railway Data Pipeline")

            if async_mode:
                # Run the async pipeline for trains
                train_locations_df, parsed_train_schedules = asyncio.run(
                    self.fetch_and_parse_all_trains_async(limit)
                )
            else:
                # Run the synchronous pipeline for trains
                train_locations_df, parsed_train_schedules = (
                    self.fetch_and_parse_all_trains(limit)
                )

            # Now fetch all lines (shapes) in one go
            line_shapes_df = self.train_data_fetcher.fetch_all_lines()

            if train_locations_df.empty and line_shapes_df.empty:
                logger.error("No data to process. Exiting.")
                return

            # Save the data
            save_format = save_format.lower()
            if save_format == "parquet":
                self.storage_handler.store_as_parquet(
                    parsed_train_schedules, line_shapes_df, train_locations_df
                )
            elif save_format == "csv":
                self.storage_handler.store_as_csv(
                    parsed_train_schedules, line_shapes_df, train_locations_df
                )
            else:
                logger.error(
                    f"Unsupported save format: {save_format}. Use 'parquet' or 'csv'."
                )
                return

            logger.success("Data pipeline completed successfully.")
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
