import csv
import time
import os
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def init_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        # Removed --disable-javascript - we need it for content loading
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(10)  # Increased for JS loading
        return driver
    except Exception as e:
        print(f"Error initializing Chrome driver: {e}")
        print("Make sure Chrome and ChromeDriver are installed and in PATH")
        raise

def scrape_page(driver, kanda, sarga, url):
    try:
        driver.get(url)
        time.sleep(2)
        
        frames = driver.find_elements(By.TAG_NAME, "frame") + driver.find_elements(By.TAG_NAME, "iframe")
        if frames:
            for frame in frames:
                try:
                    driver.switch_to.frame(frame)
                    if driver.find_elements(By.CLASS_NAME, "tat"):
                        break
                    driver.switch_to.default_content()
                except:
                    driver.switch_to.default_content()

    except Exception as e:
        print(f"Error loading page {url}: {e}")
        return []

    data = []
    current_verse_number = 0
    last_tat_index = -1

    try:
        # Get all <p> elements
        all_paragraphs = driver.find_elements(By.TAG_NAME, "p")
        
        for i, p in enumerate(all_paragraphs):
            class_attr = p.get_attribute("class")
            
            if class_attr == "pratipada":
                # Try to extract verse number from <em> inside
                try:
                    em_tag = p.find_element(By.TAG_NAME, "em")
                    em_text = em_tag.text.strip()
                    verse_match = re.search(r"^\s*(\d+)(?:[\.\-–](\d+))?", em_text)
                    if verse_match:
                        current_verse_number = verse_match.group(1)
                except:
                    pass  # No <em> or badly formatted

            elif class_attr == "tat":
                content = p.text.strip()
                if content:
                    data.append({
                        "Kanda": kanda,
                        "Sarga": sarga,
                        "Shloka": current_verse_number,
                        "Content": content
                    })
                    last_tat_index = len(data) - 1

            elif class_attr == "comment" and last_tat_index != -1:
                comment_text = p.text.strip()
                if comment_text:
                    # Append to last tat's content
                    data[last_tat_index]["Content"] += f"\n[Comment] {comment_text}"

    except Exception as e:
        print(f"Error extracting content: {e}")

    if not data:
        print(f"  WARNING: No content found on {url}")
        try:
            page_source = driver.page_source[:500]
            print(f"  Page source preview: {page_source}")
        except:
            pass

    try:
        driver.switch_to.default_content()
    except:
        pass

    return data

def main():
    input_csv = "./data/ramayana_links.csv"
    output_csv = "./data/data.csv"
    batch_size = 10

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} not found!")
        return

    driver = init_driver()
    results = []
    first_batch = True

    try:
        with open(input_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            for i, row in enumerate(reader, start=1):
                if len(row) != 3:
                    print(f"Skipping malformed row {i}: {row}")
                    continue
                    
                kanda, sarga, url = row
                print(f"Scraping {i}: {kanda} - {sarga}")
                
                try:
                    sarga_data = scrape_page(driver, kanda, sarga, url)
                    results.extend(sarga_data)
                    print(f"  Found {len(sarga_data)} items")
                except Exception as e:
                    print(f"Error scraping {url}: {e}")

                # Save in batches
                if i % batch_size == 0 and results:
                    df = pd.DataFrame(results)
                    df.to_csv(output_csv, mode='a', index=False, header=first_batch)
                    print(f"Saved batch of {len(results)} items.")
                    first_batch = False
                    results.clear()

        # Save any remaining results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, mode='a', index=False, header=first_batch)
            print(f"Saved final batch of {len(results)} items.")

    except Exception as e:
        print(f"Error reading input file: {e}")
    finally:
        driver.quit()
        print("Scraping completed.")

if __name__ == "__main__":
    main()