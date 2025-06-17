import pandas as pd
import logging
from sentence_transformers import SentenceTransformer
import spacy
import pickle
import os

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("generate_embeddings.log"),
            logging.StreamHandler()
        ]
    )

def generate_sentence_embeddings(csv_path, model_name="sentence-transformers/multi-qa-distilbert-cos-v1", output_prefix="yoga"):
    setup_logging()
    try:
        logging.info("Starting sentence embedding generation...")
        translation = {"धृतराष्ट्र": "Dhritarashtra", "सञ्जय": "Sanjay", "अर्जुन": "Arjun", "भगवान": "God", "संजय": "Sanjay"}
        nlp = spacy.load("en_core_web_sm")
        
        # Load sentence transformer model
        model = SentenceTransformer(model_name)
        logging.info(f"Loaded SentenceTransformer model: {model_name}")
        
        # Load dataset
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        data = pd.read_csv(csv_path)
        logging.info("CSV file loaded successfully.")
        
        # Generate story text
        story = ""
        for i in range(len(data)):
            if output_prefix == "gita" and data['speaker'][i] in translation:
                if translation[data['speaker'][i]] == 'Sanjay':
                    story += f" Chapter: {data['chapter'][i]}, Verse Number: {data['verse'][i]}, {translation[data['speaker'][i]]}, Narrates That, {data['translation'][i]}"
                else:
                    story += f" Chapter: {data['chapter'][i]}, Verse Number: {data['verse'][i]}, {translation[data['speaker'][i]]}, Says That, {data['translation'][i]}"
            else:
                story += f" Chapter: {data['chapter'][i]}, Verse Number: {data['verse'][i]}, {data['translation'][i]}"
        
        logging.info("Story text generated.")
        
        # Function to split text into sentences
        def text_to_sentences(text):
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        sentences = text_to_sentences(story)
        
        if not sentences:
            logging.error("No sentences extracted from the text.")
            raise ValueError("No sentences extracted from the text.")
        
        logging.info(f"Extracted {len(sentences)} sentences.")
        
        # Generate embeddings
        sentence_embeddings = model.encode(sentences, convert_to_tensor=False)
        logging.info("Sentence embeddings generated successfully.")
        
        # Save embeddings and sentences
        with open(f"sentence_embeddings_{output_prefix}.pkl", "wb") as fp:
            pickle.dump(sentence_embeddings, fp)
        with open(f"enhanced_sentences_{output_prefix}.pkl", "wb") as f:
            pickle.dump(sentences, f)
        
        logging.info("Embeddings and sentences saved.")
        return sentence_embeddings, sentences
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    embeddings, sentences = generate_sentence_embeddings("dataset/Patanjali_Yoga_Sutras_Verses_English_Questions.csv")
