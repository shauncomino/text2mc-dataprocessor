from scraper import WebScraper, WebScraperConfig
import json

def main():
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)

    web_scraper = WebScraper(scraper_config)

    web_scraper.scrape_project_links() 
    web_scraper.scrape_project_download_links()
    # web_scraper.get_image_descriptors()


if __name__ == '__main__':
    main()