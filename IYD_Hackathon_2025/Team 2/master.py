import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re
import os

# Kanda configuration
kandas = [
    {"name": "Bala Kanda", "filename": "valmiki_ramayan_bala_kanda_book1.txt", "base_url": "https://valmikiramayan.net/utf8/baala/sarga{s}/balasans{s}.htm", "total_chapters": 77},
    {"name": "Ayodhya Kanda", "filename": "valmiki_ramayan_ayodhya_kanda_book2.txt", "base_url": "https://valmikiramayan.net/utf8/ayodhya/sarga{s}/ayodhyasans{s}.htm", "total_chapters": 119},
    {"name": "Aranya Kanda", "filename": "valmiki_ramayan_aranya_kanda_book3.txt", "base_url": "https://valmikiramayan.net/utf8/aranya/sarga{s}/aranyasans{s}.htm", "total_chapters": 75},
    {"name": "Kishkindha Kanda", "filename": "valmiki_ramayan_kishkindha_kanda_book4.txt", "base_url": "https://valmikiramayan.net/utf8/kish/sarga{s}/kishkindhasans{s}.htm", "total_chapters": 67},
    {"name": "Sundara Kanda", "filename": "valmiki_ramayan_sundara_kanda_book5.txt", "base_url": "https://valmikiramayan.net/utf8/sundara/sarga{s}/sundarasans{s}.htm", "total_chapters": 68},
    {"name": "Yuddha Kanda", "filename": "valmiki_ramayan_yuddha_kanda_book6.txt", "base_url": "https://valmikiramayan.net/utf8/yuddha/sarga{s}/yuddhasans{s}.htm", "total_chapters": 128}
]

# data directory
os.makedirs("data", exist_ok=True)

# Supplementary file configuration
supplementary_file = {
    "filename": "valmiki_ramayan_supplementary_knowledge.txt",
    "url": "https://raw.githubusercontent.com/elio1278/ramayana-texts/main/valmiki_ramayan_supplementary_knowledge.txt"
}

# Function to download supplementary file
def download_supplementary_file(filename, url):
    print(f"\n📥 Downloading: {filename}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        file_path = f"data/{filename}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Downloaded: {file_path}")
        return True
    except requests.RequestException as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

# Function to scrape one kanda
def scrape_kanda(book_name, filename, base_url, total_chapters):
    verses_data = []
    content_dict = defaultdict(list)

    print(f"\n📖 Scraping: {book_name}...")

    for chapter_num in range(1, total_chapters + 1):
        url = base_url.format(s=str(chapter_num))
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            tat_paragraphs = soup.find_all("p", class_="tat")

            for verse_idx, p in enumerate(tat_paragraphs, 1):
                verse_text = p.get_text(strip=True)
                verse_number = None

                # Try previous sibling
                prev_sibling = p.find_previous_sibling()
                if prev_sibling:
                    prev_text = prev_sibling.get_text(strip=True)
                    number_match = re.search(r'(\d+)', prev_text)
                    if number_match:
                        verse_number = number_match.group(1)

                # Try beginning of verse text
                if not verse_number:
                    number_match = re.match(r'^(\d+)[\.\s]', verse_text)
                    if number_match:
                        verse_number = number_match.group(1)
                        verse_text = re.sub(r'^\d+[\.\s]*', '', verse_text)

                if not verse_number:
                    verse_number = str(verse_idx)

                content_key = verse_text.strip()
                content_dict[content_key].append((chapter_num, verse_number))

                verses_data.append({
                    'Book': book_name,
                    'Chapter': chapter_num,
                    'Verse_Number': verse_number,
                    'Content': verse_text,
                    'Content_Key': content_key
                })

            if chapter_num % 10 == 0:
                print(f"✅ Processed {chapter_num}/{total_chapters} chapters")

        except requests.RequestException as e:
            print(f"❌ Failed Chapter {chapter_num}: {e}")

    # Remove duplicates
    final_data = []
    processed_content = set()

    for verse in verses_data:
        content_key = verse['Content_Key']

        if content_key in processed_content:
            continue

        occurrences = content_dict[content_key]

        if len(occurrences) > 1:
            chapters = [str(occ[0]) for occ in occurrences]
            verse_numbers = [str(occ[1]) for occ in occurrences]
            final_data.append({
                'Book': book_name,
                'Chapter': ', '.join(chapters),
                'Verse_Number': ', '.join(verse_numbers),
                'Content': verse['Content']
            })
        else:
            final_data.append({
                'Book': book_name,
                'Chapter': verse['Chapter'],
                'Verse_Number': verse['Verse_Number'],
                'Content': verse['Content']
            })

        processed_content.add(content_key)

    # Save individual CSV for each kanda
    df = pd.DataFrame(final_data)
    csv_path = f"data/{filename.replace('.txt', '.csv')}"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"📄 Saved CSV: {csv_path}")

    # Save individual TXT for each kanda
    txt_path = f"data/{filename}"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{book_name}\n")
        f.write("=" * 50 + "\n\n")
        for _, row in df.iterrows():
            f.write(f"Chapter: {row['Chapter']}\n")
            f.write(f"Verse: {row['Verse_Number']}\n")
            f.write(f"Content: {row['Content']}\n")
            f.write("-" * 40 + "\n\n")
    print(f"📄 Saved TXT: {txt_path}")

    print(f"📊 {book_name}: {len(df)} unique verses, {len(df[df['Chapter'].astype(str).str.contains(',')])} duplicates\n")

    return df

# Main execution
print("🚀 Starting Valmiki Ramayana scraping...")

# First, download the supplementary knowledge file
download_supplementary_file(supplementary_file["filename"], supplementary_file["url"])

# Scrape all kandas
all_kanda_dataframes = []

for kanda in kandas:
    # Scrape the kanda and get the DataFrame
    df = scrape_kanda(kanda["name"], kanda["filename"], kanda["base_url"], kanda["total_chapters"])
    all_kanda_dataframes.append(df)

# Create ONLY master CSV (no master TXT)
print("\n📚 Creating master CSV file...")

# Combine all DataFrames and save master CSV
master_df = pd.concat(all_kanda_dataframes, ignore_index=True)
master_csv_path = "data/Valmiki_Ramayana_Master.csv"
master_df.to_csv(master_csv_path, index=False, encoding='utf-8')
print(f"🧾 Master CSV saved: {master_csv_path}")

print("\n✨ Scraping completed successfully!")
print(f"📊 Total verses collected: {len(master_df)}")

# Final file summary
print("\n📁 FILES GENERATED:")
print("📄 TXT Files (7 total):")
print(f"  ✅ data/{supplementary_file['filename']}")
for kanda in kandas:
    print(f"  ✅ data/{kanda['filename']}")

print("\n📄 CSV Files (7 total):")
for kanda in kandas:
    csv_name = kanda['filename'].replace('.txt', '.csv')
    print(f"  ✅ data/{csv_name}")
print(f"  ✅ data/Valmiki_Ramayana_Master.csv")

print(f"\n🎯 Total files: 14 (7 TXT + 7 CSV)")
print(f"📁 All files saved in 'data/' directory")