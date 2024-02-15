from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
import time


""" Returns list of all project links scraped """
def scrape_project_links(driver, url, file_path):
    buffer = []
    new_links_found = 0
    existing_df = initialize_dataframe(file_path)
    existing_links = set(existing_df['URL'].tolist())

    NUM_PAGES = 10 # Number of pages to scrape 
    PAGES_PER_CSV_UPDATE = 1 # Number of pages to scrape before updating the csv

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
                existing_df.loc[len(existing_df.index)] = [project_link] 
            
            buffer = [] 
            save_dataframe(existing_df, file_path)

        # Set URL to next page 
        pagination_next = driver.find_element(By.CLASS_NAME, "pagination_next")
        url = pagination_next.get_attribute("href")

    print(f"Scraped {new_links_found} new project links.")

    return list(existing_links)
    

""" (IN WORK) Scrapes download links from existing project links """
def scrape_project_download_links(driver, project_links): 

    for project_link in project_links:

        # Navigate to project page using previously scraped link 
        driver.get(project_link) 
    
        # Get project download button and slice download link from the button title
        download_button = driver.find_element(By.CLASS_NAME, 'third-party-download branded-download')
        download_button_title = download_button.get_attribute("title")
        download_link = download_button_title[29: len(download_link)]
        print("download at: " + download_link)

def initialize_browser():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment if you don't need a browser GUI
    service = Service()  # Update with your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def initialize_dataframe(file_path):
    """Initializes a DataFrame from a CSV file or creates a new one if the file doesn't exist."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['URL'])
        df.to_csv(file_path, index=False)  # Save instantly if created new
    return df


def save_dataframe(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)


def main():
    base_url = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    driver = initialize_browser()
    file_path = os.path.join(os.path.abspath('.'), 'projects.csv')
    
    project_links = scrape_project_links(driver, base_url, file_path)
    driver.quit()
    print(f"Scraped {len(project_links)} project links so far.")

if __name__ == "__main__":
    main()