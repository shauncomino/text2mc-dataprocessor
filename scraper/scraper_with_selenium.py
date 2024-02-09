from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pandas as pd
import os
import time

def scrape_project_links(driver, base_url):
    driver.get(base_url)
    project_links = []

    while True:
        # Wait for the page to load
        time.sleep(0.1)  # Small delay to mimic human-like interaction
        projects = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/project"]')

        for project in projects:
            url = project.get_attribute('href')
            if url and url not in project_links:
                project_links.append(url)
                print(url)
        
        # Find the 'Next Page' button and click it if it exists
        next_buttons = driver.find_elements(By.CSS_SELECTOR, 'a.pagination_next')
        if next_buttons:
            next_page_button = next_buttons[-1]  # Get the last button in case there are multiple
            driver.execute_script("arguments[0].click();", next_page_button)
        else:
            print("No 'Next Page' button found. End of pagination reached.")
            break

        time.sleep(0.1)  # Wait before loading the next page

    return project_links

def initialize_browser():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment if you don't need a browser GUI
    service = Service()  # Update with your ChromeDriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def main():
    base_url = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    driver = initialize_browser()
    project_links = scrape_project_links(driver, base_url)
    driver.quit()

    # Save the links to a DataFrame and then to a CSV file
    file_path = os.path.join(os.path.abspath('.'), 'projects.csv')
    df = pd.DataFrame(project_links, columns=['URL'])
    df.to_csv(file_path, index=False)
    print(f"Scraped {len(df)} project links.")

if __name__ == "__main__":
    main()
