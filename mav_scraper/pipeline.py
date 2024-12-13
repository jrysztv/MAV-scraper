from mav_scraper.requester.request_train_data import HungarianRailwayScraperPipeline


def main():
    pipeline = HungarianRailwayScraperPipeline()

    # Run the scraper workflow
    pipeline.run(
        limit=None, save_format="parquet", async_mode=True
    )  # Set `limit=None` to process all trains


if __name__ == "__main__":
    main()
