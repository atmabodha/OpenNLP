#################################################################################################################################
"""
                                       VALMIKI-RAMAYANA-VERSES SCRAPER
Brief outline of the algorithm: 
- Link Collection: Gathers chapter links by crawling through each book’s index and frame structure.

- Metadata Extraction: Extracts book number, name, and chapter number using regex.

- Pattern Matching: Filters only the <p class="tat"> elements that follow valid sequences like:
    verloc → SanSloka → SanSloka → pratipada → tat
    verloc → SanSloka → pratipada → tat
- Scan for matching sequences
    Slide a window over the tag sequence (length 5 and 4).
    For each window, check if the sequence of classes matches any valid pattern.

- Verse Processing: Extracts verse numbers from pratipada (with fallback to tat), cleans the verse text, and stores metadata.

- Output: Exports results to both .json and .csv.

- Efficiency: Uses tqdm progress bars and modular functions to track progress.
"""

import requests
from bs4 import BeautifulSoup
import re
import roman
import json
import csv
from tqdm import tqdm
import time
import unicodedata

def parsed_tree(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    return soup

# returns the list of chapters of each book in a list
def capture_bookChapter_links(link):
    parsed_data = parsed_tree(link)

    books = parsed_data.select('ol li a')
    books_link = [link['href'] for link in books]

    # stores links of books/kanda in list
    chapters_link_list = []
    with tqdm(total=len(books_link), desc="Processing Books") as pbar:
        for link in books_link:
            book_url = url + "/" + link

            parsed = parsed_tree(book_url)
            chapters = parsed.select('table td a[href]') # select the <a> tag with links
            chapters_link = [link['href'] for link in chapters]
            with tqdm(total=len(chapters_link), desc="Capturing chapter links") as cbar:
                for link in chapters_link:
                    # correcting the chapter url link
                    chapter_url = re.split("/", book_url)
                    chapter_url = url + "/" + "/".join(chapter_url[3:-1]) + "/" + link

                    parsed = parsed_tree(chapter_url)
                    chapter_url_update = parsed.select_one("frame")
                    chapter_url_src = chapter_url_update["src"]
                    chapter_url_update = url + "/" + "/".join(re.split("/", chapter_url)[3:-1]) + "/" + chapter_url_src
                    chapters_link_list.append(chapter_url_update)

                    cbar.update(1)
                    cbar.refresh()
            
            pbar.update(1)
            pbar.refresh()
                
        return chapters_link_list

# Extract Book Name function
def extract_book_name(book_str):
    pattern = r"Book\s+\w+\s*[:.]\s*(.*?Kanda)(?:\s*-|\s(?=[A-Z]))" # pattern to match Book name like Sundara Kanda
    bookname_match = re.search(pattern, book_str, flags=re.IGNORECASE | re.DOTALL)
    book_name = bookname_match.group(1).strip() if bookname_match else None
    return re.sub(r'[\x00-\x1F\x7F]', '', book_name) # remove escape character(\t, \n, \r)

# Extract Book number function
def extract_book_number(book_str):
    pattern = r'Book\s+([IVXLCDM]+)\s*[:.]' # pattern to match the Book number like 1, 2, 3, etc
    booknumber_match = re.search(pattern, book_str, re.IGNORECASE)
    book_number = roman.fromRoman(booknumber_match.group(1).strip()) if booknumber_match else None
    return book_number  

# Extract Chapter number
def extract_chapter_number(book_str):
    pattern = r"(?:Chapter\s+)?(?:\[\s*Sarga\s*\]\s*)?(\d+)" # pattern to match the chapter number like 1, 19, 3, 55, etc
    chapter_number_match = re.search(pattern, book_str, re.IGNORECASE)
    chapter_number = int(chapter_number_match.group(1)) if chapter_number_match else None
    return chapter_number

# Extract verse number
def extract_from_pratipada(pratipada):
    pattern = r'^\s*((?:\d+[a-zA-Z]?)(?:[-;,]\s*\d+[a-zA-Z]?)*)(?:\.)?'
    nums = re.findall(pattern, pratipada)
    return nums
    
def extract_from_tat(verse_txt):
    pattern = r"\[\d+-\d+-([^\]]+)\]"
    verse_number_match = re.search(pattern, verse_txt)
    if verse_number_match:
        nums = verse_number_match.group(1)
        return [v.strip() for v in re.split(r'[;,]', nums)]
    return []

# Normalize the verse text
def normalize(text):
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # removes the verse marker ([?-?-?])
    text = re.sub(r'\[\s*\d+(?:\s*-\s*\d+){0,2}[a-zA-Z]?(?:\s*[\s.,-]\s*[a-zA-Z]?\d*[a-zA-Z]?)*\s*]?\s*', '', text)
        
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text 

# function which return the final list data structure in suitable format
def main(lst):
    verse_list = []
    with tqdm(total=len(lst), desc="Capturing chapter links") as pbar:
        for link in lst:
            r4 = requests.get(link)
            soup4 = BeautifulSoup(r4.content, 'html.parser')
        
            # pre processing book text
            book_text = soup4.select("h3")
            book_str = ""
            for i in range(len(book_text)):
                book_str += book_text[i].get_text(strip=True)
            
            book_number = extract_book_number(book_str)
            book_name = extract_book_name(book_str)
            chapter_number = extract_chapter_number(book_str)
        
            tags = soup4.find_all("p")
            # Define the pattern sequence
            patterns = [
                ["verloc", "SanSloka", "SanSloka", "pratipada", "tat"], # or
                ["verloc", "SanSloka", "pratipada", "tat"]
            ] 
            
            # Convert tags into list of (class, text)
            tag_classes = [(tag.get("class")[0], tag.get_text(strip=True)) for tag in tags if tag.get("class")]
            
            # Extract "tat" values that follow the exact pattern
            with tqdm(total=len(soup4.find_all('p', class_='tat')), desc=f'Verses in Chapter {chapter_number}', leave=False) as verse_bar:
                for i in range(len(tag_classes)):
                    for pattern in patterns:
                        plen = len(pattern)
                        if i + 1 >= plen: # Ensure enough tags behind
                            sequence = [tag_classes[i - plen + j + 1][0] for j in range(plen)]
                            verse_map = {}
                            if sequence == pattern and pattern[-1] == 'tat':
                                tat_text = tag_classes[i][1].replace('\r', '').replace('\n', ' ').replace('\t', '') # i points to 'tat'
                        
                                # extract from pratipada class
                                pratipada_txt = tag_classes[i - 1][1].replace('\r', '').replace('\n', ' ')
                                if pratipada_txt:
                                    verse_numbers = extract_from_pratipada(pratipada_txt)
                                    if not verse_numbers:
                                        # Fall back to tat class extraction
                                        verse_numbers = extract_from_tat(tat_text)
                                    
                                # text normalizing
                                tat_text = normalize(tat_text)
                                
                                # here if verse_numbers is empty or contain empty string -> skip
                                if '' not in verse_numbers and verse_numbers:
                                    # assign different attributes
                                    verse_map['book_name'] = book_name
                                    verse_map['book_number'] = book_number
                                    verse_map['chapter_number'] = chapter_number
                    
                                    # assign verse and verse number
                                    verse_map['verse_number'] = verse_numbers
                                    verse_map['verse'] = tat_text
                                    verse_marker = ",".join(verse_numbers)
                                    verse_map['verse_id'] = f"[{book_number}-{chapter_number}-{verse_marker}]"
                                    
                                    verse_list.append(verse_map)
                    
                                verse_bar.update(1) 
                                verse_bar.refresh()
                    
            pbar.update(1)
            pbar.refresh()
    
    return verse_list

# Convert to json string
def json_convertor(data):
    json_data = json.dumps(data, ensure_ascii=False, indent=2)

    # save to a file
    json_file = 'valmiki-ramayana-verses.json'
    with open(json_file, 'w', encoding="utf-8") as f:
        f.write(json_data)

    print(f"Saved {len(data)} entries to {json_file}")

# Save as CSV file
def csv_convertor(data):
    # Header: Kanda/Book	Sarga/Chapter	Shloka/Verse Number	English Translation

    # rename Keys for CSV header
    csv_data = []
    for entry in data:
        csv_row = {
            "Kanda/Book": entry["book_name"],
            "Sarga/Chapter": entry["chapter_number"],
            "Shloka/Verse Number": ", ".join(entry["verse_number"]),  # convert list to string
            "English Translation": entry["verse"]
        }

        csv_data.append(csv_row)

    # Write to CSV
    csv_file = "valmiki-ramayana-verses.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Saved {len(csv_data)} entries to {csv_file}")

def convertTime(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

if __name__ == "__main__":
    url = "https://valmikiramayan.net" 
    main_page = "/ramayana.html"
    link = url + main_page

    start_time = time.time()

    chapters_link_lst = capture_bookChapter_links(link)
    result = main(chapters_link_lst)
    
    # Save file to JSON and CSV
    json_convertor(result)
    csv_convertor(result)

    end_time = time.time()
    execution_time = convertTime(end_time - start_time)

    print(f"Execution time: {execution_time}")