from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from openai import OpenAI
import time
import os
from dataclasses import dataclass, field
from typeguard import typechecked

# Control csv save rate 
PAGES_PER_CSV_UPDATE = 1; 
DOWNLOAD_LINKS_PER_CSV_UPDATE = 5

# Map downloading global variables
is_downloading_first_time = True
file_name = ""

# 25 links per full page 
PAGES_TO_SCRAPE = 2


@dataclass
class WebScraperConfig:
    def default_csv_columns():
        return ["PAGE_URL", "DOWNLOAD_URL", "IMAGE_URL", "GPT4_DESCRIPTION", "AUTHOR_DESCRIPTION", "FILE_TYPE", "BUILD_PATH"]
    
    PROJECT_DESCRIPTION_PROMPT: str = None
    """ ChatGPT-4 prompt for generating image descriptions. """

    OPEN_AI_API_KEY: str = None
    """ OpenAI API key. """

    CSV_FILE_PATH: str = None
    """ CSV file path."""

    CSV_COLUMNS: list = field(default_factory=default_csv_columns)
    """ Default columns for the CSV file. """

    BASE_URL: str = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    """ Base planetminecraft.com URL """

    BUILD_DOWNLOAD_DIRECTORY: str = None
    """ Directory to save the build downloads. """


    def initialize_browser(self):
        chrome_options = Options()
        chrome_options.enable_downloads = True
        #chrome_options.add_argument("--headless")  # Uncomment if you don't need a browser GUI
        chrome_options.add_argument('log-level=3') # Only log "fatal" errors (most aren't actually program-critical)
        
        prefs = {
                "download.default_directory" : self.BUILD_DOWNLOAD_DIRECTORY,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
            }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)

        return driver

    def __post_init__(self):
        if self.OPEN_AI_API_KEY is None:
            raise ValueError("OpenAI API key is required.")
        
        if self.BUILD_DOWNLOAD_DIRECTORY is None:
            self.BUILD_DOWNLOAD_DIRECTORY = os.path.join(os.path.abspath('.'), 'builds')

        self.openai_client = OpenAI(api_key=self.OPEN_AI_API_KEY)
        
        if not self.CSV_FILE_PATH or not os.path.exists(self.CSV_FILE_PATH):
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
        self.gpt4_prompt = self.cfg.PROJECT_DESCRIPTION_PROMPT
    
    """ Saves the dataframe to a csv file."""
    def save_to_csv(self):
        self.projects_df.to_csv(self.cfg.CSV_FILE_PATH, index=False)

    """Scrapes project and image links"""
    def scrape_project_links(self):
        num_new_links = 0

        current_url = self.cfg.BASE_URL 

        # Iterate for given number of pages 
        for project_pages_scraped in range (0, PAGES_TO_SCRAPE):
            self.driver.get(current_url) # load next page  

            # Pull list of r-info classes from page
            r_info_classes = self.driver.find_elements(By.CLASS_NAME, 'r-info')
            image_urls = self.driver.find_elements(By.CSS_SELECTOR, 'picture')

            # Look at each r-info class in list 
            for index in range (0, len(r_info_classes)):
                # Scrape project page link from the class 
                new_url = (r_info_classes[index].find_element(By.CSS_SELECTOR, 'a[href^="/project"]')).get_attribute("href") 
                image_url = image_urls[index].find_element(By.CSS_SELECTOR, 'img').get_attribute("src")

                target_row = self.projects_df[self.cfg.CSV_COLUMNS[0]] == new_url
                project_url_column = self.cfg.CSV_COLUMNS[0]

                # Check if the new link is original 
                if new_url and not self.projects_df.loc[target_row, project_url_column].any():
                    num_new_links += 1
                    self.projects_df.loc[len(self.projects_df)] = [new_url, "", image_url, "", "", "", ""]
            
            # Update csv 
            if (project_pages_scraped % PAGES_PER_CSV_UPDATE == 0): 
                self.save_to_csv()

            # Update current page 
            pagination_next = self.driver.find_element(By.CLASS_NAME, "pagination_next")
            current_url = pagination_next.get_attribute("href")
        
        print(f"Scraped {num_new_links} new project urls.")


    """Scrapes the project download links using previously captured project page links"""
    def scrape_project_download_links(self): 
        download_links_scraped = 0

        for row in range (0, len(self.projects_df.index)): 
            download_url = self.projects_df.loc[row, self.cfg.CSV_COLUMNS[1]] 

            if download_url == "" or pd.isna(download_url):
                project_link = self.projects_df.loc[row, self.cfg.CSV_COLUMNS[0]]
            
                # Navigate to project page using previously scraped link 
                self.driver.get(project_link) 
            
                # Get internal Planet Minecraft download link
                internal_download_link = self.get_internal_download_link()

                # Get third party download link
                third_party_download_link = self.get_third_party_download_link()

                # Check and store download links 
                if internal_download_link:
                    self.projects_df.loc[row, self.cfg.CSV_COLUMNS[1]] = internal_download_link
                    download_links_scraped += 1
                elif third_party_download_link:
                    self.projects_df.loc[row, self.cfg.CSV_COLUMNS[1]] = third_party_download_link
                    download_links_scraped += 1
                else: 
                    print("No project link found for " + project_link) 
                
                # Update csv 
                if (download_links_scraped % DOWNLOAD_LINKS_PER_CSV_UPDATE == 0): 
                    self.save_to_csv()

        print(f"Scraped {download_links_scraped} new download urls.")

    def update_csv(self, row):
        global file_name

        file_type = ".{0}".format(file_name.split('.')[-1])
        local_file_path = f"builds/{file_name}"

        # TODO: Create a new tuple and replace to current row
        # self.projects_df.loc[row.index, row.FILE_TYPE] = file_type
        # self.projects_df.loc[row.index, row.BUILD_PATH] = local_file_path
        # self.save_to_csv()

    """ Go through the CSV file and download each map """
    def download_project_maps(self):
        for row in self.projects_df.itertuples():
            row_download_url = row.DOWNLOAD_URL

            # Check internal download link
            if "planetminecraft.com" in row_download_url:
                self.download_internal_map(row_download_url)
            # Otherwise external download link
            else:
                self.download_external_map(row_download_url)
                continue # TODO: Remove this "continue" after finishing the external map download function

            # Update CSV file
            self.update_csv(row)

    """ Scrape a download link for third party websites """
    def get_third_party_download_link(self):
        try:
            download_button = self.driver.find_element(By.CLASS_NAME, 'third-party-download')
            download_button_hover_text = download_button.get_attribute('title').split(' ')
            download_button_link = download_button_hover_text[-1]
    
            return download_button_link
        except:
            return None
        
    """ Scrape a download link from the internal Planet Minecraft website """
    def get_internal_download_link(self):
        try:
            download_button = self.driver.find_element(By.CLASS_NAME, 'branded-download')
            download_button_href = download_button.get_attribute('href')

            # Check if download link is external
            if "mirror" in download_button_href:
                raise Exception
            return download_button_href
        except:
            return None

    """ Gets a build description using ChatGPT-4 """
    def get_build_descriptions(self): 
        descriptions_generated = 0
    
        for row in range (0, len(self.projects_df.index)):
            build_description = self.projects_df.loc[row, self.cfg.CSV_COLUMNS[3]]

            if build_description == "" or pd.isna(build_description):
                new_description = self.gpt4_image_prompt(self.projects_df.loc[row, self.cfg.CSV_COLUMNS[2]])
                self.projects_df.loc[row, self.cfg.CSV_COLUMNS[3]] = new_description 
                descriptions_generated += 1
            
            if (descriptions_generated % DOWNLOAD_LINKS_PER_CSV_UPDATE == 0):
                self.save_to_csv()

    """ Send ChatGPT-4 a message including an image"""
    def gpt4_image_prompt(self, image_url): 
        response = self.openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": self.gpt4_prompt}, 
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

        # Remove quotes, newlines, and leading/trailing whitespace
        return response.choices[0].message.content.strip().replace('\n', '').replace("\"", '')
    
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
            time.sleep(0.33)

            # Close second tab
            print("Closing second tab")
            self.driver.close()
            time.sleep(0.33)

            # Switch to original tab
            print("Switching to first tab")
            self.driver.switch_to.window(self.driver.window_handles[0])
            time.sleep(0.33)

    def wait_until_download_finished(self):
        # Go to the downloads page in Chrome
        self.driver.get('chrome://downloads/')

        # While the map is not downloaded
        while True:
            time.sleep(1)

            # Check if the map is downloading (Chrome shows a pause and cancel buttons)
            try:
                pause_button = self.driver.find_element(By.ID, 'pauseOrResume')
            # Else the map finished downloading (There is no more pause and cancel buttons)
            except:
                # Update the file name
                global file_name
                file_name = self.driver.execute_script("return document.querySelector('downloads-manager').shadowRoot.querySelector('#downloadsList downloads-item').shadowRoot.querySelector('div#content  #file-link').text")
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

    # TODO: Add support for external map downloads
    def download_external_map(self, external_download_link):
        print("External download link found")
