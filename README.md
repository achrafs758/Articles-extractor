# Articles-extractor

## Description:

Articles-extractor is a project designed to extract figures and tables, along with their page numbers and bounding boxes, from PDF documents. The extracted data is presented in a tabular format using Streamlit for easy visualization.

### Current Development Status:

The project is under active development, with a focus on addressing the following key issues:

1. **Handling Multi-Block Pages:** Enhancements are being made to handle pages with multiple blocks of content, ensuring accurate extraction of figures and tables.

2. **Improving Layout Model and OCR Accuracy:** Ongoing efforts are dedicated to improving the accuracy of the layout model and Optical Character Recognition (OCR) for precise figure and table extraction.

## Project Structure:

The project comprises the following components:

1. **PDF Extraction:** Utilizes a PDF parsing module to extract content, bounding boxes, and page numbers from PDF documents.

2. **Layout Model:** Employs a layout model to identify figures and tables based on their formatting and layout.

3. **OCR (Optical Character Recognition):** Applies OCR to extract captions and labels associated with figures and tables.

4. **Streamlit Web Application:** Displays extracted figures and tables, along with corresponding page numbers and bounding boxes, in tabular format using Streamlit, providing an interactive user interface.

## Requirements:

Ensure you have the following installed:

- Python 3.7 or higher
- Libraries: pandas, numpy, streamlit, Deepdoctection, Detectron2, Pandas, CGBoost, joblib, NLTK, XGBoost

## Installation Guide:

### Install Python:

1. Download and install Python 3.7 or a higher version from [Python's official website](https://www.python.org/downloads/).

### Install Docker (Optional):

1. Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) to install Docker.

### Clone the Repository:

1. Open a terminal or command prompt.

2. Run the following command to clone the repository:

    ```bash
    git clone https://github.com/your-username/Articles-extractor.git
    ```

### Install Project Dependencies:

1. Navigate to the project directory:

    ```bash
    cd Articles-extractor
    ```

2. Run the following command to install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage with Docker:

1. Run the following command to build the Docker image:

    ```bash
    docker-compose build
    ```

2. After the build is complete, use the following command to run the application:

    ```bash
    docker-compose up
    ```

3. Access the application in your web browser at `http://localhost:8501`.

4. Upload your PDF documents to the application.

5. The application will display tables containing figures and tables, along with their respective page numbers and bounding boxes.

## Note:

Make sure the files uploaded are in the same folder as the application.

## Author:

Soulala Achraf | 
Email: achrafs758@gmail.com

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
