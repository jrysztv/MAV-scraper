# MAV-scraper-analysis
Scraping train location data for educational purposes. Automated with GitHub actions. Data is collected directly in the repo and stored in .parquet format.
Dataset location: /MAV-SCRAPER/data/parquet_store

Data included:
- parsed_train_shapes.parquet - Shapes data for each train passing (needs work to drop duplicates and efficiently store)
- parsed_train_schedules.parquet - The schedule data for each train, containing reported expected and actual arrival and departure times
- spot_train_locations.parquet - The real-time location of trains at a given time

This is the first step of a train delay analytical pipeline that leverages the graph structure of train data.

Here is an in-depth Medium article on how it was done:
https://medium.com/p/25cc2111cb3c
