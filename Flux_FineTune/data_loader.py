from datasets import load_dataset
from huggingface_hub import login
import os


# Set your Hugging Face API token as an environment variable
os.environ["HF_API_TOKEN"] = "Your HF Token"

# Log in to Hugging Face using the token stored in the environment variable
login(token=os.environ["HF_API_TOKEN"])

# Load the dataset from a specified directory (replace "you data dir" with your actual directory path)
dataset = load_dataset("imagefolder", data_dir="you data dir", drop_labels=True)

# Push your dataset to the Hugging Face Hub using your dataset ID
dataset.push_to_hub("Your HF dataset ID")
# Example: dataset.push_to_hub("Nitin9sitare/stable_diffusion_Swami_Vivekananda")

