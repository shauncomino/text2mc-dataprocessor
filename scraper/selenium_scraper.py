from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from openai import OpenAI
import os
from dataclasses import dataclass
import inspect

from typing import List, Dict, typechecked
import json

# Control csv save rate 
PAGES_PER_CSV_UPDATE = 1; 
DOWNLOAD_LINKS_PER_CSV_UPDATE = 5

# Flag to check if the user started to download maps
is_downloading_first_time = True

# 25 links per full page 
PAGES_TO_SCRAPE = 2

# TODO: Read from file
API_KEY = ""

@dataclass
class WebScraperConfig:
    OPEN_AI_API_KEY: str = None
    """ OpenAI API key. """

    CSV_FILE_PATH: str = None
    """ CSV file path."""

    CSV_COLUMNS: list = ["PAGE_URL", "DOWNLOAD_URL", "IMAGE_URL", "GPT4_DESCRIPTION", "AUTHOR_DESCRIPTION", "FILE_TYPE", "BUILD_PATH"]
    """ Default columns for the CSV file. """

    BASE_URL: str = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    """ Base planetminecraft.com URL """

    BUILD_DOWNLOAD_DIRECTORY: str = None
    """ Directory to save the build downloads. """

    def initialize_browser():
        chrome_options = Options()
        chrome_options.enable_downloads = True
        #chrome_options.add_argument("--headless")  # Uncomment if you don't need a browser GUI
        chrome_options.add_argument('log-level=3') # Only log "fatal" errors (most aren't actually program-critical)

        service = Service()  # Update with your ChromeDriver path
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def __post_init__(self):
        if self.OPEN_AI_API_KEY is None:
            raise ValueError("OpenAI API key is required.")
        
        if self.BUILD_DOWNLOAD_DIRECTORY is None:
            self.BUILD_DOWNLOAD_DIRECTORY = os.path.join(os.path.abspath('.'), 'builds')

        self.openai_client = OpenAI(self.OPEN_AI_API_KEY)

        if not self.CSV_FILE_PATH or not os.path.exists(self.CSV_COLUMNS):
            self.df = pd.DataFrame(columns=self.CSV_COLUMNS)
        else: 
            self.df = pd.read_csv(self.CSV_FILE_PATH)

        
        self.driver = self.initialize_browser()
        

class WebScraper:
    @typechecked
    def __init__(self, cfg: WebScraperConfig):
        self.cfg = cfg
        self.driver = self.cfg.driver
        self.projects_df = self.cfg.df
        self.openai_client = self.cfg.openai_client
        
    """Scrapes project and image links"""
    def scrape_project_links(self, url):
        num_new_links = 0

        # Iterate for given number of pages 
        for project_pages_scraped in range (1, PAGES_TO_SCRAPE):
            self.driver.get(url) # load URL 

            # Pull list of r-info classes from page
            r_info_classes = self.driver.find_elements(By.CLASS_NAME, 'r-info')
            pictures = self.driver.find_elements(By.CSS_SELECTOR, 'picture')

            index = 0
            # Look at each r-info class in list 
            for r_info_class in r_info_classes:
                # Scrape project page link from the class 
                new_link = (r_info_class.find_element(By.CSS_SELECTOR, 'a[href^="/project"]')).get_attribute("href") 
            
                # Make sure don't already have link
                if new_link and len(self.projects_df['PAGE_URL']) == 0:
                    num_new_links += 1
                    image_url = pictures[index].find_element(By.CSS_SELECTOR, 'img').get_attribute("src")
                    self.projects_df.loc[len(self.projects_df)] = [new_link, "", image_url, ""]
                    index += 1
                    print("Found new link: " + str(new_link))
            
            # Update csv 
            if (project_pages_scraped % PAGES_PER_CSV_UPDATE == 0): 
                save_to_csv(dict_to_df(data_dict), file_path)

            # Update URL
            pagination_next = self.driver.driver.find_element(By.CLASS_NAME, "pagination_next")
            url = pagination_next.get_attribute("href")

        print(f"Scraped {num_new_links} new project links.")

    """ Scrape a download link from the internal Planet Minecraft website """
    def get_internal_download_link(self):
        try:
            download_button = self.driver.find_element(By.CLASS_NAME, 'branded-download')
            download_button_href = download_button.get_attribute('href')

            # Check if download link is external
            if "mirror" in download_button_href:
                raise Exception

            print("Internal Download Link Found:", download_button_href)
            return download_button_href
        except:
            print("No Internal Download Link Found")
            return None

    def get_image_descriptors(self, data_dict): 
        prompt = "Caption this Minecraft build like you are making a dataset for a generative ML mod"
        project_links = list(data_dict.keys())

        descriptions_generated = 0
        for project_link in project_links:
            hasDescriptors = type(data_dict.get(project_link).get(DESCRIPTORS)) == str

            if not hasDescriptors: 
                descriptors = self.gpt4_image_prompt(prompt, data_dict.get(project_link).get(IMAGE_URL))

                data_dict.get(project_link).update({DESCRIPTORS: descriptors})
                descriptions_generated += 1
            
            if (descriptions_generated % DOWNLOAD_LINKS_PER_CSV_UPDATE == 0):
                self.save_to_csv(self.dict_to_df(data_dict), self.CSV_FILE_PATH)

    """Gets a response back from ChatGPT-4 for a prompt including an image"""
    def gpt4_image_prompt(self, prompt, image_url): 
        response = self.client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}, 
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
        )
        return response.choices[0].message.content.strip().replace('\n', '')

    """ Scrape a download link for third party websites """
    def get_third_party_download_link(self):
        try:
            download_button = self.driver.find_element(By.CLASS_NAME, 'third-party-download')
            download_button_hover_text = download_button.get_attribute('title').split(' ')
            download_button_link = download_button_hover_text[-1]
            print("Third-Party Download Link Found:", download_button_link)
            return download_button_link
        except:
            print("No Third Party Download Link Found")
            return None

    """Scrapes the project download links using previously captured project page links"""
    def scrape_project_download_links(self, data_dict, file_path): 
        project_links = list(data_dict.keys())
        download_links_scraped = 0
        for project_link in project_links:
            hasDownloadLink = type(data_dict.get(project_link).get(DOWNLOAD_URL)) == str
            if not hasDownloadLink: 
                download_links_scraped += 1
                # Navigate to project page using previously scraped link 
                self.driver.get(project_link) 
            
                # Get internal Planet Minecraft download link
                internal_download_link = self.get_internal_download_link()

                # Get third party download link
                third_party_download_link = self.get_third_party_download_link()

                # Check and store download links 
                if internal_download_link:
                    data_dict.get(project_link).update({DOWNLOAD_URL: internal_download_link})
                    #data_dict.update({project_link: {DOWNLOAD_URL: internal_download_link, DESCRIPTORS: None}})
                elif third_party_download_link:
                    data_dict.get(project_link).update({DOWNLOAD_URL: third_party_download_link})
                    #data_dict.update({project_link: {DOWNLOAD_URL: third_party_download_link, DESCRIPTORS: None}})
                else: 
                    print("No project link found for " + project_link) 
                
                # Update csv 
                if (download_links_scraped % DOWNLOAD_LINKS_PER_CSV_UPDATE == 0): 
                    self.save_to_csv(self.dict_to_df(data_dict), file_path)

        print(f"Scraped {download_links_scraped} new download links.")

    """ Clicking the download button for the first time opens a sponsor waiting page """
    """ For the first time downloading, click the download button and close the sponsor page """
    def handle_first_map_download(self):
        print("First time downloading")

        global is_downloading_first_time
        is_downloading_first_time = False

        # Click the download button
        download_button = self.driver.find_element(By.CLASS_NAME, 'branded-download')
        self.driver.execute_script("arguments[0].click()", download_button)

        # Wait until the sponsor page tab opens
        wait = WebDriverWait(self.driver, 10)
        wait.until(EC.number_of_windows_to_be(2))

        # Two tabs open: (1) the original map page and (2) the sponsor waiting page
        if len(self.driver.window_handles) == 2:
            # Switch to second tab
            print("Switching to second tab")
            self.driver.switch_to.window(self.driver.window_handles[1])

            # Close second tab
            print("Closing second tab")
            self.driver.close()

            # Switch to original tab
            print("Switching to first tab")
            self.driver.switch_to.window(self.driver.window_handles[0])

    def wait_until_download_finished(self):
        # Go to the downloads page in Chrome
        self.driver.get('chrome://downloads/')

        # While the map is not downloaded
        while True:
            # Check if the map is downloading (Chrome shows a pause and cancel buttons)
            try:
                pause_button = self.driver.find_element(By.ID, 'pauseOrResume')
            # Else the map finished downloading (There is no more pause and cancel buttons)
            except:
                break

    def download_internal_map(self, internal_download_link):
        self.driver.get(internal_download_link)
        map_title = self.driver.find_element(By.ID, 'resource-title-text').text
        print("Downloading map:", map_title)

        global is_downloading_first_time
        
        if is_downloading_first_time:
            self.handle_first_map_download()

        # Download the map
        download_button = self.driver.find_element(By.CLASS_NAME, 'branded-download')
        self.driver.execute_script("arguments[0].click()", download_button)

        # Wait until the download finishes
        self.wait_until_download_finished()
        print("Finished downloading:", map_title)

    """Converts dataframe to dictionary, with the project page links being the unique keys"""
    def df_to_dict(self): 
        return self.df.set_index(PAGE_URL).T.to_dict('dict')

    """Converts dictionary to dataframe"""
    def dict_to_df(self): 
        page_links = list(data_dict.keys())
        df = pd.DataFrame.from_dict(data_dict, orient='index', columns=[PAGE_URL, DOWNLOAD_URL, IMAGE_URL, DESCRIPTORS])
        df[PAGE_URL] = page_links
        return df
        
    """Initializes a DataFrame from a CSV file or creates a new one if the file doesn't exist."""
    def initialize_dataframe(self):
        if not os.path.exists(self.cfg.CSV_FILE_PATH):
            df = pd.DataFrame(columns=[PAGE_URL, DOWNLOAD_URL, IMAGE_URL, DESCRIPTORS])

        df = pd.read_csv("projects.csv")
        return df

    """Saves the DataFrame to a CSV file."""
    def save_to_csv(self):
        df.to_csv(self.cfg.CSV_FILE_PATH, index=False)

def main():
    
    driver = initialize_browser()
    file_path = os.path.join(os.path.abspath('.'), 'projects.csv')
    data_dict = df_to_dict(initialize_dataframe(file_path))
    
    scrape_project_links(driver, base_url, data_dict, file_path) 
    scrape_project_download_links(driver, data_dict, file_path)
    get_image_descriptors(data_dict, file_path)

    driver.quit()

if __name__ == "__main__":
    main()