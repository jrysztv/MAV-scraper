from mav_scraper.requester.request_train_data import MAVScraper

if __name__ == "__main__":
    scraper = MAVScraper()

    # Run the scraper workflow
    scraper.run(limit=None)  # Set `limit=None` to process all trains
