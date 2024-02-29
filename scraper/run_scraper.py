from selenium_scraper import WebScraper, WebScraperConfig
import json

def main():
    config_path = "C:/text2mc/SCRUM-15/text2mc-dataprocessor/scraper/webscraper_config.json"
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    web_scraper.scrape_project_links() 
    web_scraper.scrape_project_download_links()
    web_scraper.get_build_descriptions()
    web_scraper.download_project_maps()

if __name__ == '__main__':
    main()