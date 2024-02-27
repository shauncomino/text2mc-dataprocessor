from selenium import webdriver
import time


driver = webdriver.Chrome()

driver.get("https://www.planetminecraft.com/projects/land-structure/?platform=1&monetization=0&share=world_link&order=order_latest")

time.sleep(30)

driver.quit()