from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
import time

PAGES_PER_CSV_UPDATE = 1; 
DOWNLOAD_LINKS_PER_CSV_UPDATE = 5
NUM_PAGES = 2

is_downloading_first_time = True

""" Returns list of all project links scraped """
def scrape_project_links(driver, url, file_path):
    buffer = []
    new_links_found = 0
    existing_df = initialize_dataframe(file_path)
    existing_links = set(existing_df['URL'].tolist())

    # NUM_PAGES = 200 # Number of pages to scrape 
    # PAGES_PER_CSV_UPDATE = 1 # Number of pages to scrape before updating the csv

    for i in range (1, NUM_PAGES):
        # Navigate to page 
        print("\nNavigating to: " + url)
        driver.get(url)
        time.sleep(0.1) 

        # Get project links from page 
        projects = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/project"]')

        # Get original links from 
        for project in projects:
            url = project.get_attribute('href')
            if url and url not in existing_links:
                new_links_found += 1
                existing_links.add(url)
                buffer.append(url)
                print("Found new link: " + url)
        
        # To prevent major data loss in case of network error, update CSV periodically 
        if (i % PAGES_PER_CSV_UPDATE == 0): 

            # Append project links in buffer to dataframe
            for project_link in buffer: 
                existing_df.loc[len(existing_df.index), "PROJECT PAGE URL"] = project_link
            
            buffer = [] 
            save_dataframe(existing_df, file_path)

        # Set URL to next page 
        pagination_next = driver.find_element(By.CLASS_NAME, "pagination_next")
        url = pagination_next.get_attribute("href")

    print(f"Scraped {new_links_found} new project links.")

    return list(existing_links)

""" Scrape a download link from the internal Planet Minecraft website """
def get_internal_download_link(driver):
    try:
        download_button = driver.find_element(By.CLASS_NAME, 'branded-download')
        download_button_href = download_button.get_attribute('href')

        # Check if download link is external
        if "mirror" in download_button_href:
            raise Exception

        print("Internal Download Link Found:", download_button_href)
        return download_button_href
    except:
        print("No Internal Download Link Found")
        return None

""" Scrape a download link for third party websites """
def get_third_party_download_link(driver):
    try:
        download_button = driver.find_element(By.CLASS_NAME, 'third-party-download')
        download_button_hover_text = download_button.get_attribute('title').split(' ')
        download_button_link = download_button_hover_text[-1]
        print("Third-Party Download Link Found:", download_button_link)
        return download_button_link
    except:
        print("No Third Party Download Link Found")
        return None

""" (IN WORK) Scrapes download links from existing project links """
def scrape_project_download_links(driver, project_links, file_path): 
    buffer = []
    existing_df = initialize_dataframe(file_path)
    
    for i in range (0, len(project_links)):
        project_link = project_links[i]
        
        # Navigate to project page using previously scraped link 
        driver.get(project_link) 
    
        # Get internal Planet Minecraft download link
        internal_download_link = get_internal_download_link(driver)

        # Get third party download link
        third_party_download_link = get_third_party_download_link(driver)

        if internal_download_link:
            buffer.append(internal_download_link)
        
        elif third_party_download_link:
            buffer.append(third_party_download_link)
        
        else: 
            print("No project link found for " + project_link) 
            buffer.append("")

        if (len(buffer) == DOWNLOAD_LINKS_PER_CSV_UPDATE): 
            for link in buffer:
                existing_df.loc[i, "PROJECT DOWNLOAD URL"] = link
            
            buffer = [] 
            save_dataframe(existing_df, file_path)

def wait_until_download_finished():
    time.sleep(10)

def download_internal_map(driver, internal_download_link):
    global is_downloading_first_time

    driver.get(internal_download_link)

    # Clicking the download button for the first time opens a sponsor waiting page
    # For the first time downloading, click the download button and close the sponsor page
    if is_downloading_first_time:
        print("First time downloading")
        is_downloading_first_time = False

        # Click the download button
        download_button = driver.find_element(By.CLASS_NAME, 'branded-download')
        driver.execute_script("arguments[0].click()", download_button)

        # Wait until the sponsor page tab opens
        wait = WebDriverWait(driver, 10)
        wait.until(EC.number_of_windows_to_be(2))

        # Two tabs open: (1) the original map page and (2) the sponsor waiting page
        if len(driver.window_handles) == 2:
            # Switch to second tab
            print("Switching to second tab")
            driver.switch_to.window(driver.window_handles[1])

            # Close second tab
            print("Closing second tab")
            driver.close()
            
            # Switch to original tab
            print("Switching to first tab")
            driver.switch_to.window(driver.window_handles[0])

    # Download the map
    print("Downloading map")
    download_button = driver.find_element(By.CLASS_NAME, 'branded-download')
    driver.execute_script("arguments[0].click()", download_button)

    # Wait until the download finishes
    wait_until_download_finished()
    print("Map downloaded")

def initialize_browser():
    chrome_options = Options()
    chrome_options.enable_downloads = True
    # chrome_options.add_argument("--headless")  # Uncomment if you don't need a browser GUI
    service = Service()  # Update with your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def initialize_dataframe(file_path):
    """Initializes a DataFrame from a CSV file or creates a new one if the file doesn't exist."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['PROJECT PAGE URL', 'PROJECT DOWNLOAD URL'])
        df.to_csv(file_path, index=False)  # Save instantly if created new
    return df


def save_dataframe(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)


def main():
    base_url = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    driver = initialize_browser()
    file_path = os.path.join(os.path.abspath('.'), 'projects.csv')
    
    # project_page_links = scrape_project_links(driver, base_url, file_path); 

    # Test scraping links
    # scrape_project_download_links(driver, project_page_links, file_path)

    # project_links = scrape_project_links(driver, base_url, file_path)

    internal_test_map = "https://www.planetminecraft.com/project/the-moon-5763469/download/worldmap/"
    download_internal_map(driver, internal_test_map)

    driver.quit()
    # print(f"Scraped {len(project_links)} project links so far.")

if __name__ == "__main__":
    main()