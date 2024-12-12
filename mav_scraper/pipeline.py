from mav_scraper.requester.request_train_data import HungarianRailwayScraperPipeline


def main():
    pipeline = HungarianRailwayScraperPipeline()

    # Run the scraper workflow
    pipeline.run(limit=None)  # Set `limit=None` to process all trains


if __name__ == "__main__":
    main()
