from selenium_scraper import WebScraper, WebScraperConfig
import json
import os

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webscraper_config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    test_mediafire_link = "https://www.mediafire.com/file/uyoi29ruip5v3dy/United_States_tjc%25286%2529.zip/file"
    test_googledrive_link = ""

    web_scraper.download_external_map(test_mediafire_link)


if __name__ == '__main__':
    main()