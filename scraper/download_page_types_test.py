from selenium_scraper import WebScraper, WebScraperConfig
import json
import os

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webscraper_config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    test_link = "https://static.planetminecraft.com/files/resource_media/schematic/boulevardier-s-shophouse.zip"

    web_scraper.download_with_raw_link(test_link)
    
if __name__ == '__main__':
    main()