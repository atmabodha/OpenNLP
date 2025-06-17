<div align="center">
  <span style="font-size:22px; font-weight:bold;">🧠 Truth Retrieval: Context-Aware Fact-Checking with Language Models</span><br>
  <span style="font-size:16px;"><i>An End-to-End System for Verifying Claims Using Ancient Texts</i></span>
</div>
</br>

---

An end-to-end fact-checking system for verifying claims against **structured textual sources** such as ancient epics like the **Valmiki Ramayana**. The system adopts a **Hybrid Retrieval-Augmented Generation (Hybrid RAG)** approach that integrates modern Information Retrieval (IR) techniques—both dense (semantic) and sparse (lexical)—with Language Models (LLMs) for contextual understanding and reasoning. By combining these components, the system retrieves the most relevant verses, re-ranks them based on semantic alignment, and generates a well-grounded verdict (True, False, or Irrelevant) with supporting evidence and explanation.

---

## 📌 Use Case
Given a claim (e.g., *“Hanuman was a minister of Sugriva”*), the system:
1. Retrieval: Identifies the most relevant verses from the structured corpus using both **dense semantic retrieval (via FAISS)** and **sparse keyword-based retrieval (via BM25)**.
2. Rank Fusion: Combines the results from both retrieval methods using **Reciprocal Rank Fusion (RRF)** to create a unified, ranked list of candidate verses.
3. Re-ranking: Evaluates each candidate verse in relation to the claim using a **Cross-Encoder** (e.g., `ms-marco-MiniLM-L6-v2`), which jointly considers the claim and verse for deeper contextual alignment.
3. Verification: Passes the top-ranked, contextually aligned verses to an **LLM-based verifier** (`Mistral-7B-Instruct-v0.2`) that determines whether the claim is **True**, **False**, or **Irrelevant**, along with supporting evidence and reasoning.

---

## 🛠️ Pipeline Overview

### 1. 🚧 Dataset Construction
#### 1.1 🏪 Data Collection and Storage
- Data is scraped from the official translation at https://valmikiramayan.net. The data is saved in a structured CSV format.
- Each verse entry includes `book/kanda name, chapter/sarga and verse/shloka numbers, verse text (in English)`. 

#### 1.2 🔎 Text Preprocessing
- Preprocess verse texts and user's claim by lowering, stripping extra white space and removing unnecessary punctiations.
- Applied **Spelling Variant Mapping** (standardizes names) to ensure robust matching across variations e.g.,  `"sita" <-> "seetha", "lakshmana" <-> "lakshman"`, etc.

### 2. 🧭 Dual Retrieval
- **Dense Retrieval (FAISS + Inner Product)**:
   - Convert **all scraped verses** and user’s **natural language claim** into a dense vector using a Sentence Transformer (e.g., `intfloat/e5-large-v2`).
   - Performs fast semantic similarity search using FAISS (IP index).
   - Returns top-k matches ranked by cosine similarity.

- **Sparse Retrieval (BM25)**:
  - Traditional keyword-based retrieval using `rank_bm25`.
  - Returns top-k matches based on term overlap and inverse document frequency.

### 3. 🔁 Rank Fusion (RRF)
- Combine both FAISS and BM25 ranked lists using **Reciprocal Rank Fusion**:
$$
\text{RRF}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}
$$

- Where:
   - `d` is a document (or verse, in this context),
   - `rank_i(d)` is the rank position of document d in the i-th ranked list (e.g., from dense or sparse retrieval),
   - `k` is a constant (typically set to 60) that reduces the influence of lower-ranked items.

- Final top-100 ranked candidates are a balanced mix of semantic and lexical relevance.

### 4. 🧮 Cross-Encoder Reranking
- Re-rank top 100 retrieved verses by scoring **claim + verse pairs** using a **CrossEncoder** (e.g., `ms-marco-MiniLM-L6-v2`) model.
- The cross-encoder sees the claim and verse jointly and understands their contextual relationship.
- Returns the top 10 re-ranked verses.

### 5. ✅ Final Verdict Generation
Use a lightweight LLM (e.g., **Mistral-7B-Instruct**) to:
- Analyze top 10 re-ranked verses.
- Determine if they support, contradict, or are irrelevant to the claim.
- Output a structured JSON:
```json
{
   "relevance": "RAMAYANA_RELATED" or "NOT_RAMAYANA_RELATED",
   "label": "TRUE" or "FALSE",
   "confidence_score": 1-10,
   "reference": ["Bala Kanda 66 13b, 14a, ..."],
   "explanation": "Reasoning behind the verdict"
}
```

---

## ⚙️ Technology Stack

| Component        | Description                                 |
|------------------|---------------------------------------------|
| **Web Scraper** |	Web scraping and structured data extraction from online verse sources (e.g., `BeautifulSoup4`)|
| **SentenceTransformer**  | Embedding claim and verse (e.g., `intfloat/e5-large-v2`)  |
| **FAISS**        | Dense retrieval with Inner Product search   |
| **BM25**         | Sparse keyword-based retrieval              |
| **RRF**          | Fuses dense and sparse retrieval rankings   |
| **CrossEncoder** | Re-ranking candidate verses for relevance (e.g., `ms-marco-MiniLM-L6-v2`)  |
| **LLM**   | Context-aware judgment & explanation  (e.g., `Mistral-7B-Instruct-v0.2`)      |
| **JSON API**     | Outputs structured and explainable results  |

---

## 📊 Why This Architecture?

| Design Choice        | Justification |
|----------------------|---------------|
| **Dual Retrieval**   | Covers both semantic & keyword-based signals |
| **RRF Fusion**       | Avoids bias toward one retrieval method |
| **CrossEncoder**     | Deep semantic matching beyond shallow vector similarity |
| **Light LLM**        | Balances performance and accuracy for verdict explanation |

---

## 🔬 Example

**Claim**: "On seeing Hanuman, Sita lost her consciousness for a long time."

→ Hybrid RAG retrieves top 10 re-ranked verses:
   1. `as soon as seeing hanuman sita lost her consciousness very much and became seemingly lifeless. regaining her consciousness after a long time the wide eyed sita moreover thought (as follows)`

   2. `on seeing her for a moment, hanuman ascertained her as sita and became dejected. that sita was indeed seen by him, not long ago`

   3. `even after seeing the mighty hanuman who came there, sita kept herself silent. then, seeing and recollecting him, she became rejoiced`

   4. `... 7... 10.`
  
→ LLM interprets and explains (Output):

```json
{
   "relevance": "RAMAYANA_RELATED",
   "label": "TRUE",
   "confidence_score": 0.9,
   "reference": [
      (1, 'Sundara Kanda, Sarga 32, Shloka 8'), (4, 'Sundara Kanda, Sarga 32, Shloka 1'), 
      (6, 'Sundara Kanda, Sarga 34, Shloka 12'), (10, 'Sundara Kanda, Sarga 40, Shloka 20;21')
   ],
   "explanation": "The textual evidence shows that Sita lost consciousness or had deep sighs upon seeing Hanuman multiple times. (Sundara Kanda, Sarga 32, Shloka 8; Sundara Kanda, Sarga 32, Shloka 1; Sundara Kanda, Sarga 34, Shloka 12; Sundara Kanda, Sarga 40, Shloka 20;21)"
}
```

---

## 🚀 Setup Instructions

## 1. Access the Project Directory

```bash
cd ramayana_fact_checker-by_pradyuman
```

## 2. (Recommended) Create and Activate a Virtual Environment
Using Python 3.9.13 (preferred for this project, else whats available can work too):

<details> <summary><strong>💻 On Linux/macOS:</strong></summary>

```bash
python3.9 -m venv venv
source venv/bin/activate
```
</details> 

<details> <summary><strong>On Windows (if multiple Python versions are registered):</strong></summary>

```bash
py -3.9 -m venv venv
venv\Scripts\activate
```
</details>

✅ Make sure Python 3.9.13 is installed and available in your environment. You can check with python --version

## 3. Install dependencises

```bash
pip install -r requirements.txt
```

## 3. Run the app
- To extract the verse data:
   ```bash
   python scraper.py
   ```

- To run prediction on Test data:
   ```bash
   python factchecker-pipeline.py
   ```
   - Specify the file name of Test data in line 413 of file `factchecker-pipeline.py`.
      <details> <summary>Example code block (lines 411 - 417)</summary>

      ```python
      if __name__ == '__main__':
         # Load test data
         df_test = pd.read_csv('Valmiki Ramanaya Verses - Final Evaluation.csv') # change file name as necessary
         results = []
         
         # Mistral Model call
         verifier = ClaimVerifier()
      ```

      </details>

   - Here current test data file name is `Valmiki Ramanaya Verses - Final Evaluation.csv`, change it to your CSV file name.
---

## 📂 File Structure
```bash
.
├── factchecker-pipeline.py      # To run predictions as a python file                
├── factchecker-pipeline.ipynb   # To run predictions as a python notebook
├── scraper.py                   # Data Extraction code file
├── requirements.txt             # Required dependencies
├── valmiki-ramayana-verses.csv  # Structured verse corpus
└── Readme.md                    # Markdown script for pipeline explanation and how to get app running

```
---

## 📝 Customizing Output
If you want to control what output gets saved into the final CSV (`RamanayaFinal.csv`), you can modify the script accordingly:
```
💡 To save only the desired output to the CSV, comment out the relevant lines in the script.
```

### Instructions
In `factchecker-pipeline.py`, comment out the lines in 474 to 496 in append method to exclude the ones that you don't require:

<details> <summary>Example code block (lines 474–496):</summary>

```python
            results.append({
                #"ID": row["ID"],
                "Statement": claim,
                #"Truth": row['Truth'],
                "Prediction": predicted_truth,
                #"Reference": row['Reference'],
                #"Predicted_Reference": predicted_ref,
                "Explanation": model_xplanation
            })

            print(f"PREDICTION: {predicted_truth}")
            print(f"Explanation: {model_xplanation}")
            print("====================")
        else:
            results.append({
                #"ID": row["ID"],
                "Statement": claim,
                #"Truth": row['Truth'],
                "Prediction": 'NOT RELEVANT',
                #"Reference": np.nan,
                #"Predicted_Reference": np.nan,
                "Explanation": 'Not relevant to the Ramayana verses'
            })
```

</details>

- For now I have kept only the `Statement`, `Prediction` and `Explanation` columns. These will be the same columns that will be reflected in `RamanayaFinal.csv` file.
- Columns like `ID`, `Truth` and `Reference` are test file dependent. If these are availbale in test files then only they will work otherwise don't comment out (will caluse error).

---

# 📧 Contact

For bug or error reports and feature requests related to FactChecker, please write an email to pradyuman.thakur@gmail.com

---