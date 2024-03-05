from selenium_scraper import WebScraper, WebScraperConfig
import json
import os

def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webscraper_config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    scraper_config = WebScraperConfig(**config)
    web_scraper = WebScraper(scraper_config)

    test_link = "https://download1514.mediafire.com/y3k7sx6z5log6jKY89T8QaqhDKy59KF-TdeZ133J8PEYrsC44Zss_iFXXivPKWZN6F9Hw9vFoaAdJhDTIMo3WaHxWOy2ttA80sThZ2AjLqt2afRNFZ1OdTMlfuQTBaym2ajJHL2rZSybcsIwSSFdTwo3JW_2gyq6hxQ85hsC3Zyz4Q/qy4qfwk23z9ao9a/FarmHouseLandscape.zip"

    web_scraper.download_with_raw_link(test_link)


if __name__ == '__main__':
    main()