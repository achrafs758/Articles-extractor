# Articles-extractor

**Description:**

This project aims to extract figures and tables along with their page numbers and bounding boxes from PDF documents. The extracted data is then presented in tabular format using Streamlit for easy visualization.

**Note:**
Currently, the project is under development, and there are two main issues being addressed:

1. **Handling Two Blocks on a Page:** The current implementation assumes a single block of content on each page. However, some articles may span multiple blocks across pages. The code is being enhanced to handle such cases and ensure accurate extraction of figures and tables.

2. **Improving Layout Model and OCR Accuracy:** The accuracy of the layout model and OCR (Optical Character Recognition) is crucial for precise figure and table extraction. Ongoing efforts are being made to improve the layout model and OCR accuracy to achieve better results.

**Project Structure:**

The project consists of the following components:

1. **PDF Extraction:** A PDF parsing module is used to extract the content, bounding boxes, and page numbers from the PDF documents.

2. **Layout Model:** A layout model is employed to identify figures and tables in the extracted content based on their formatting and layout.

3. **OCR (Optical Character Recognition):** OCR is applied to extract captions and labels associated with figures and tables.

4. **Streamlit Web Application:** The extracted figures and tables along with their corresponding page numbers and bounding boxes are displayed in tabular format using Streamlit, providing an interactive user interface.

**Requirements:**

- Python 3.7 or higher
- Libraries: pandas, numpy, streamlit, Deepdoctection,Detectron2,Pandas, CGBoost, joblib, NLTK, 
- XGBoost
- joblib


**Instructions:**

1. Install the required libraries using `pip install -r requirements.txt`.

2. Run the Streamlit application using `streamlit run app.py`.

3. Upload your PDF documents to the application.

4. The application will display a table containing figures and another table containing tables, along with their respective page numbers and bounding boxes.

**Note:** The project is continuously being improved, and updates will be made to address the aforementioned issues and enhance the overall functionality and accuracy of the figure and table extraction process. Feedback and contributions are welcome.
Make sure the files uploaded à re in the same folder as 

**Author:**
Soulala Achraf | 
achrafs758@gmail.com

**License:**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
