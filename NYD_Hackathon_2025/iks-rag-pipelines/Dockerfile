FROM --platform=linux/amd64 continuumio/miniconda3:latest

# Set working directory
WORKDIR /.

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
RUN conda create -n myenv python=3.11 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install dependencies using conda first
RUN conda install -c conda-forge \
    numpy=1.24.3 \
    spacy=3.7.2 \
    thinc=8.1.12 \
    pandas \
    scipy \
    -y

# Install pytorch
RUN conda install -c pytorch \
    pytorch \
    -y

# Install sentence-transformers and other packages
RUN pip install --no-cache-dir \
    sentence-transformers \
    transformers \
    scikit-learn \
    groq \
    qdrant-client \
    better-profanity \
    nltk

# Copy requirements and install remaining packages
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --no-deps

COPY . .

EXPOSE 8000

# Run using the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 