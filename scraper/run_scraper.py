from selenium_scraper import WebScraper, WebScraperConfig
import json
import os


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "webscraper_config.json"
    )
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    # web_scraper.scrape_project_links(pages_to_scrape=1770)
    # web_scraper.scrape_project_page_info()
    web_scraper.scrape_raw_map_download_links()
    # web_scraper.download_all_builds()


if __name__ == "__main__":
    main()
