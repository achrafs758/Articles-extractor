# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libb2-dev \
    libpangocairo-1.0-0 \
    libglib2.0-0 \
    build-essential \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    software-properties-common \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project directory into the container
COPY . .

# Install the main required Python packages
RUN pip install -r requirements.txt

# Install NLTK data
RUN python -m nltk.downloader punkt stopwords

# Clone the GitHub repository and install the component
RUN git clone https://github.com/deepdoctection/deepdoctection.git && \
    cd deepdoctection && \
    pip install ".[source-pt]" && \
    cd .. && \
    rm -rf deepdoctection

# Expose port 8501
EXPOSE 8501

# Define a health check for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Specify the entry point for the container
ENTRYPOINT ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]
