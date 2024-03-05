from selenium_scraper import WebScraper, WebScraperConfig
import json
import os

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webscraper_config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    web_scraper.scrape_project_links() 
    web_scraper.scrape_project_download_links()
    # web_scraper.get_build_descriptions()
    web_scraper.scrape_raw_map_download_links()

if __name__ == '__main__':
    main()