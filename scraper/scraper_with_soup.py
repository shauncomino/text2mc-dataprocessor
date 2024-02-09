import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

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

def scrape_project_links(base_url, file_path, headers):
    """Scrapes project links and updates the DataFrame instantly when new links are found."""
    page = 1
    existing_df = initialize_dataframe(file_path)
    existing_links = set(existing_df['URL'].tolist())
    
    while True:
        new_links_found = False
        print(f"Scraping page: {page}")
        response = requests.get(f'{base_url}&p={page}', headers=headers)
        
        if response.status_code != 200:
            print(f"Stopped due to invalid response: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        project_list = soup.find('ul', class_='resource_list grid project')
        
        if not project_list:
            print("No project list found on the page.")
            break
        
        for tag in project_list.find_all('a', href=True):
            url = tag['href']
            if url.startswith('/project'):
                full_url = f"https://www.planetminecraft.com{url}"
                if full_url not in existing_links:
                    existing_links.add(full_url)
                    existing_df = existing_df._append({'URL': full_url}, ignore_index=True)
                    new_links_found = True
        
        if new_links_found:
            save_dataframe(existing_df, file_path)  # Save instantly when new links are found
            print(f"Page {page}: New links found. DataFrame updated and saved.")
        else:
            print(f"Page {page}: No new links found.")
        
        page += 1
        time.sleep(.3)

def main():
    base_url = "https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'projects.csv')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    scrape_project_links(base_url, file_path, headers)
    print("Scraping complete.")

if __name__ == "__main__":
    main()
