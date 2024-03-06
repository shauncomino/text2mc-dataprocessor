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
from typing import Optional
import requests
from tqdm import tqdm
import re
import traceback


# Control csv save rate 
PAGES_PER_CSV_UPDATE = 5; 
DOWNLOAD_LINKS_PER_CSV_UPDATE = 5
ROWS_EDITED_PER_UPDATE = 100


@dataclass
class WebScraperConfig:
    def default_csv_columns():
        return ["PAGE_URL", "DOWNLOAD_URL", "IMAGE_URL", "GPT4_DESCRIPTION", "TAGS", "RAW_DOWNLOAD_LINK"]
    
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

    NO_GUI: Optional[int] = 0
    """ Whether or not the browser should be headless """

    def initialize_browser(self):
        chrome_options = Options()
        chrome_options.enable_downloads = True

        if self.NO_GUI:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument('log-level=3') # Only log "fatal" errors (most aren't actually program-critical)
        
        prefs = {
                "download.default_directory" : self.BUILD_DOWNLOAD_DIRECTORY,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "profile.default_content_settings.cookies": 2,
                "profile.block_third_party_cookies": True

            }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)

        return driver

    def __post_init__(self):
        if self.OPEN_AI_API_KEY is None:
            raise ValueError("OpenAI API key is required.")
        
        if self.BUILD_DOWNLOAD_DIRECTORY is None or not os.path.exists(self.BUILD_DOWNLOAD_DIRECTORY):
            self.BUILD_DOWNLOAD_DIRECTORY = os.path.join(os.path.abspath('./'), 'builds')

        os.makedirs(self.BUILD_DOWNLOAD_DIRECTORY, exist_ok=True)
        self.openai_client = OpenAI(api_key=self.OPEN_AI_API_KEY)
        
        if not self.CSV_FILE_PATH or not os.path.exists(self.CSV_FILE_PATH):
            self.df = pd.DataFrame(columns=self.CSV_COLUMNS)
            self.CSV_FILE_PATH = os.path.join(os.path.abspath('./'), "projects.csv")
            if (os.path.exists(self.CSV_FILE_PATH)):
                print("Previous CSV file found! Loading...")
                self.df = pd.read_csv(self.CSV_FILE_PATH)
                self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
                print(self.df.head())
        else: 
            print("Previous CSV file found! Loading...")
            self.df = pd.read_csv(self.CSV_FILE_PATH)
            self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
            print(self.df.head())
        
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
    def scrape_project_links(self, pages_to_scrape: Optional[int] = 1):
        num_new_links = 0

        current_url = self.cfg.BASE_URL 

        # Iterate for given number of pages 
        for project_pages_scraped in range (0, pages_to_scrape):
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

                if new_url and not (self.projects_df[project_url_column] == new_url).any():
                    num_new_links += 1
                    # Create a dictionary with all column names initialized to an empty string or appropriate default value
                    row_data = {col: "" for col in self.projects_df.columns}
                    # Assign new_url to the "PAGE_URL" column and image_url to the "IMAGE_URL" column
                    row_data["PAGE_URL"] = new_url
                    row_data["IMAGE_URL"] = image_url
                    # Append the new row to the DataFrame
                    self.projects_df.loc[len(self.projects_df)] = row_data
                            
            # Update csv 
            if (project_pages_scraped % PAGES_PER_CSV_UPDATE == 0): 
                self.save_to_csv()

            # Update current page 
            pagination_next = self.driver.find_element(By.CLASS_NAME, "pagination_next")
            current_url = pagination_next.get_attribute("href")
        
        print(f"Scraped {num_new_links} new project urls.")
        self.save_to_csv()


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
                    self.projects_df.loc[row, "DOWNLOAD_URL"] = internal_download_link
                    download_links_scraped += 1
                elif third_party_download_link:
                    self.projects_df.loc[row, "DOWNLOAD_URL"] = third_party_download_link
                    download_links_scraped += 1
                else: 
                    print("No project link found for " + project_link) 
                
                # Update csv 
                if (download_links_scraped % DOWNLOAD_LINKS_PER_CSV_UPDATE == 0): 
                    self.save_to_csv()

        print(f"Scraped {download_links_scraped} new download urls.")

    """ Go through the CSV file and scrape the raw download links """
    def scrape_raw_map_download_links(self):
        for index, row in self.projects_df.iterrows():
            row_download_url = row["DOWNLOAD_URL"]
            raw_download_link = ""

            # Check internal download link
            if "planetminecraft.com" in row_download_url:
                raw_download_link = self.scrape_internal_raw_download_link(row_download_url)
            # Otherwise external download link
            elif "mediafire" in row_download_url:
                raw_download_link = self.scrape_third_party_raw_download_link(row_download_url)
            else:
                pass
            
            # Add the raw download link to the CSV file
            if raw_download_link is not None:
                self.projects_df.loc[index, "RAW_DOWNLOAD_LINK"] = raw_download_link
                
            if index % ROWS_EDITED_PER_UPDATE == 0:
                self.save_to_csv()
        
        self.save_to_csv()

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
    
    """ Scrapes the tags of all the existing builds in the dataframe """
    def scrape_tags(self):
        for index, row in self.projects_df.iterrows():
            try:
                page_url = row["PAGE_URL"]
                tags = self.scrape_tags_of_one_build(page_url)

                if len(tags) > 0:
                    self.projects_df.loc[index, "TAGS"] = tags
                
                if index % ROWS_EDITED_PER_UPDATE == 0:
                    self.save_to_csv()
            except Exception as e:
                print(e)
                print(f"Failed to scrape tags for link: {page_url}")
            
        self.save_to_csv()


    """ Gets the tags of a single build from the build page """
    def scrape_tags_of_one_build(self, build_page_url):
        tags_list = list()
        self.driver.get(build_page_url)
        tags = self.driver.find_elements(By.CLASS_NAME, "tag")
        for tag in tags:
            # Extracting the inner text of the tag element
            tag_text = tag.find_element(By.TAG_NAME, "a").text
            # Appending the extracted text to the tags_list
            tags_list.append(tag_text)
            
        return str(tags_list)
    
    """ Gets the GET requestable URL from the planetminecraft build page using regular expressions """
    def scrape_internal_raw_download_link(self, build_page_url):
        download_link = None
        try:
            self.driver.get(build_page_url)
            scripts = self.driver.find_elements(By.TAG_NAME, "script")
            for script in scripts:
                if "schematic:" in script.get_attribute("innerHTML"):
                    # Extract the URL using regular expression
                    matches = re.search(r'schematic: "(.*?)",', script.get_attribute("innerHTML"), re.DOTALL)
                    if matches:
                        s3_url = matches.group(1)
                        # Extract the part of the URL after 'static.planetminecraft.com' and before the query parameters
                        url_path = re.search(r'static\.planetminecraft\.com(.*?\.\w+)', s3_url)
                        if url_path:
                            # Form the direct download link
                            download_link = f"https://static.planetminecraft.com{url_path.group(1)}"
                            break
        except Exception as e:
            print(e)
            print("Couldn't locate direct download link for planetminecraft hosted build")
                    
        return download_link
    
    """ Yoinks the GET requestable URL from the third-party link """
    def scrape_third_party_raw_download_link(self, external_download_link):
        self.driver.get(external_download_link)

        download_button = None
        raw_download_link = None

        try:
            if "mediafire" in external_download_link:
                download_button = self.driver.find_element(By.ID, "downloadButton")
                raw_download_link = download_button.get_attribute('href')
        except:
            print("Download page not available")

        return raw_download_link
    
    """ Downloads all the builds using get requests """
    def download_all_builds(self):
        for index, row in self.projects_df.iterrows():
            print(row)
            raw_url = row["RAW_DOWNLOAD_LINK"]

            # Check if raw_url is not NaN, not None, and not an empty string
            if pd.notna(raw_url) and raw_url:
                try:
                    self.download_with_raw_link(raw_download_link=raw_url)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())  # This ensures that you see the traceback of the exception
                    print(f"Failed to download with link: {raw_url}")
            else:
                print("Skipping empty or invalid URL.")

            
    """ Downloads a single build using a get request """
    def download_with_raw_link(self, raw_download_link: str = None, filename: Optional[str] = None):
        # Make a request to get the file
        response = requests.get(raw_download_link, stream=True)

        # Extract filename from URL if not provided
        if filename is None:
            filename = raw_download_link.split("/")[-1]

        # Combine the directory and filename
        filepath = os.path.join(self.cfg.BUILD_DOWNLOAD_DIRECTORY, filename)

        # Total size in bytes.
        total_size = int(response.headers.get('content-length', 0))

        # Initialize the progress bar
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        # Open the file with write-binary mode
        with open(filepath, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

        # Check if the file was downloaded completely
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")


    


    