from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv

# Function to extract links for Sargas from a given Kanda page
def extract_sarga_links(driver, kanda_name, kanda_url):
    driver.get(kanda_url)
    time.sleep(2)  # wait for the page to load

    # Extract all Sarga links from the Kanda page (assuming they are in <a> tags)
    sargas = driver.find_elements(By.TAG_NAME, "a")
    
    data = []
    for sarga in sargas:
        sarga_title = sarga.text.strip()
        sarga_url = sarga.get_attribute('href')
        
        # Filter for valid Sarga links by checking if 'sarga' appears in the URL
        # and ensure the URL follows the expected structure with chapter numbers
        if sarga_title and sarga_url and 'sarga' in sarga_url.lower() and '/sarga' in sarga_url:
            # Extract chapter number if present in the URL
            chapter_number = sarga_url.split('/')[-2]  # Gets the chapter number from the URL path
            data.append([kanda_name, f"{chapter_number}", sarga_url])
    
    return data

# Initialize WebDriver
driver = webdriver.Chrome()

# Open the main page
driver.get("https://valmikiramayan.net/")
time.sleep(2)  # wait for the page to load

# Switch to the main frame where the content is
driver.switch_to.frame("main")

# Kanda sections, where we will gather the URLs of each Kanda
kanda_links = driver.find_elements(By.TAG_NAME, "a")
kandas = set()  # Set to ensure uniqueness

# Collect Kanda names and their corresponding URLs
for kanda in kanda_links:
    kanda_name = kanda.text
    kanda_url = kanda.get_attribute('href')
    
    if kanda_name and kanda_url:
        # Add Kanda name and URL as a tuple to the set (this removes duplicates automatically)
        kandas.add((kanda_name, kanda_url))

# Prepare CSV file to store the results
with open('./data/ramayana_links.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Kanda", "Sarga", "url"])

    # Loop through each Kanda and extract the Sarga links
    for kanda_name, kanda_url in kandas:
        sarga_data = extract_sarga_links(driver, kanda_name, kanda_url)
        writer.writerows(sarga_data)

# Close the WebDriver after extraction
driver.quit()

print("Link extraction complete. The links has been saved in 'ramayana_links.csv'.")
