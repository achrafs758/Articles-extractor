#streamlit app to display articles
import streamlit as st
import deepdoctection as dd
from pathlib import Path
import re
import pandas as pd
from IPython.core.display import HTML
import re
import pickle
import joblib
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

analyzer = dd.get_dd_analyzer()

st.set_page_config(layout="centered", page_title="Articles extractor", page_icon="🧮")


with st.sidebar:
    st.title("Articles extraction Application")
    st.write(":one: Pdf importation.")
    st.write(":two: Display articles and their class.")
    st.write(":three: Layouts details(Tables and Figures).")
    st.write(":four: Display tables.")
    st.write("## Notes:  \n - Once you upload the file, a \"Running\" sign with show at the top right \n - Once it has completed running, a button will appear to download the file")
#create button to upload pdf file

dt = None
if 'dt' not in st.session_state:
    st.session_state.dt = dt
else:
    dt = st.session_state.dt
#add button to start process

pages = None
if 'pages' not in st.session_state:
    st.session_state.pages = dt
else:
    pages = st.session_state.pages

with st.expander("Pdf importation.", expanded=True):
    file = st.file_uploader("Upload pdf file", type=["pdf"])
    if file is not None:
                df = analyzer.analyze(path=file.name)
                df.reset_state()  # This method must be called just before starting the iteration. It is part of the API.

                doc = iter(df)
                pages=[]
                for _ in range(len(df)):
                    page = next(doc)
                    pages.append(page)
                st.session_state.pages = pages
                                    #diplay pdf infos and content 
                #define base64 function to display pdf
                def show_pdf(file_path):
                    with open(file_path,"rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                st.write(show_pdf(file.name))
                #number of pages
                st.write("Number of pages: ", int(len(df)))
                
                stemmer = SnowballStemmer("french")
                # DataFlair - Initialize a TfidfVectorizer
                f = open('./ML_Model/tfidf_vectorizer.pkl', 'rb')
                tfidf_vectorizer = pickle.load(f)
                f.close()

                f = open('./ML_Model/model_95.81.pkl', 'rb')
                classifier_model = pickle.load(f)
                f.close()

                # Load Spacy Model
                # spacy_model = spacy.load("./spacy-model/model-best")
                
                def preprocess_text(text):
                    """This utility function sanitizes a string by:
                    - removing links
                    - removing special characters
                    - removing numbers
                    - removing stopwords
                    - transforming in lowercase
                    - removing excessive whitespaces
                    Args:
                        text (str): the input text you want to clean
                        remove_stopwords (bool): whether or not to remove stopwords
                    Returns:
                        str: the cleaned text
                    """
                    
                    text = re.sub("[^A-Za-z]+", " ", text)
                    tokens = nltk.word_tokenize(text)
                    tokens = [w for w in tokens if not w.lower() in stopwords.words("french")]
                    text = " ".join(tokens)
                    text = text.lower().strip()
                    text = ' '.join([stemmer.stem(word) for word in text.split()])
                    return text

                def predict(model, article):
                    article_preprocessed = preprocess_text(article)
                    article_preprocessed = ' '.join([stemmer.stem(word) for word in article_preprocessed.split()])
                    tfidf_article = tfidf_vectorizer.transform([article_preprocessed])
                    prediction = model.predict(tfidf_article)
                    if prediction[0] == 1:
                        return 'Zone'
                    return 'Not Zone'
                    

                def is_article_start(layout):
                    return re.match(r"^(art|ART|Art)", layout.text)

                def is_article_end(layout, next_layout):
                    return is_article_start(next_layout) if next_layout else False

                def sort_layouts_by_coordinates(layouts):
                    return sorted(layouts, key=lambda layout: (layout.bounding_box.uly, layout.bounding_box.ulx))

                def extract_article_content(layouts):
                    content = ''
                    is_title = True
                    for layout in layouts:
                        if is_title and is_article_start(layout):
                            content += layout.text + ' '
                            continue
                        is_title = False
                        if layout.category_name in ['text', 'list']:
                            content += layout.text + ' '
                        elif layout.category_name == 'title':
                            break
                    return content.strip()

                # Process each page
                article_data = []
                for page_layout in pages:
                    sorted_layouts = sort_layouts_by_coordinates(page_layout.layouts)
                    articles_start_indices = [i for i, layout in enumerate(sorted_layouts) if is_article_start(layout)]
                    articles_end_indices = [i-1 for i in articles_start_indices[1:]] + [len(sorted_layouts)-1]

                    for start, end in zip(articles_start_indices, articles_end_indices):
                        article_layouts = sorted_layouts[start:end+1]
                        article_content = extract_article_content(article_layouts)
                        if article_content:
                            article_data.append({'ID': len(article_data) + 1, 'Content': article_content,'Type': predict(classifier_model, article_content)})

                dt = pd.DataFrame(article_data)   
                st.session_state.dt = dt   
                count1=0
                count3=0
                count4=0
                count5=0
                #display count of each layout category and articles 
                for page in pages:
            #create a button to download text
                    for layout in page.layouts:
                        if layout.category_name=="title":
                            count1+=1
                        if layout.category_name=="list":
                            count3+=1
                        if layout.category_name=="figure":
                            count5+=1
                    for table in page.tables:
                        count4+=1
                st.write("Number of articles: ", int(len(dt)))
                st.write("Number of titles: ", int(count1))
                st.write("Number of lists: ", int(count3))
                st.write("Number of figures: ", int(count5))
                st.write("Number of tables: ", int(count4))
                texts=[page.text for page in pages]
                #seperate between pages using a symbol
                all_text='\n'.join(texts)
                st.download_button("⬇️ Download entire Pdf content as text",all_text, "text.txt", "text/plain", use_container_width=True)

    #if prompt contains a value


with st.expander("Display articles and their class."):
    if dt is not None:
        if st.session_state.dt is not None:
            
            def highlight_greaterthan(row,token_to_highlight=None):
                #iterate on 
                styles = ['background-color: #a24857 ' if token_to_highlight in str(row['Content']) else '' for _ in row.index]
                return styles
            dt=st.session_state.dt
            prompt=st.text_input("Enter a word to highlight in the articles")
            
            if prompt:
                st.dataframe(dt.style.apply(highlight_greaterthan,token_to_highlight=prompt, axis=1))
            else:
                st.dataframe(dt)
            st.download_button("⬇️ Download articles as .csv", dt.to_csv(), "annotated.csv", use_container_width=True)
        
    # Display the data frame
        # Process each page
        #add a section of articles,table,lists,figures count
#add secton displaying infos of figure and table layout



with st.expander("Layout details(Tables and Figures)."):
        if dt is not None:
            if st.session_state.pages is not None:
                pages=st.session_state.pages
                #table that shows figure layout details, page where it is located and it's bounding box        
                def sort_layouts_by_coordinates(layouts):
                    return sorted(layouts, key=lambda layout: (layout.bounding_box.uly, layout.bounding_box.ulx))
                layout_data = []
                for page_layout in pages:
                    sorted_layouts = sort_layouts_by_coordinates(page_layout.layouts)
                    for layout in sorted_layouts:
                        if layout.category_name=="figure" or layout.category_name=="table":
                            layout_data.append({'Page': page_layout.page_number, 'Type': layout.category_name, 'Bounding Box': layout.bounding_box})
                #if layout_data not empty
                dt2 = pd.DataFrame(layout_data)
                st.dataframe(dt2)
                st.download_button("⬇️ Download layout details as .csv", dt2.to_csv(), "layout_details.csv", use_container_width=True)
                
#display tables
with st.expander("Display tables."):
    if dt is not None:
        if st.session_state.dt is not None:
            i=0                
            for i,page in enumerate(pages):
                if page.tables:    
                    st.write("Page ",i+1)    
                    for index,table in enumerate(page.tables):
                        st.write("Table ",index+1)
                        st.write(HTML(table.html))
                        st.write("\n")
