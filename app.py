# Streamlit app to display articles
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
import concurrent.futures  # Import concurrent.futures for multithreading
import time  # Import the time module


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt','stopwords')
import tempfile  # Import tempfile to create a temporary file


st.set_page_config(layout="centered", page_title="Extracteur d'articles", page_icon="🧮")

st.markdown(
    "## Extracteur des articles: \n  Bienvenue dans l'application d'extraction d'articles. Cette application utilise un modèle "
    "**Cascade RCNN** pour extraire les articles des fichiers PDF en les segmentant. Les articles "
    "sont ensuite affichés de manière structurée. "
    "Le processus se déroule comme suit :"
)

st.markdown(
    "1. **Importation du fichier PDF :** Importez votre fichier PDF en utilisant le bouton ci-dessous. "
    "Assurez-vous que le fichier importé contient des articles que vous souhaitez extraire."
)

st.markdown(
    "2. **Segmentation des articles :** Une fois le fichier PDF importé, le modèle Cascade RCNN entre en action. "
    "Il analyse le document et identifie les zones correspondant à chaque article. Cette étape est marquée par l'affichage d'un indicateur 'En cours d'exécution' en haut à droite."
)

st.markdown(
    "3. **Affichage des articles :** Après la segmentation réussie, les articles extraits sont affichés de manière structurée. "
    "Vous pouvez parcourir et examiner chaque article individuellement."
)

st.markdown(
    "###### Remarques:  \n - Une fois que vous avez importé le fichier, un signe \"En cours d'exécution\" apparaîtra en haut à droite. "
    "Cela indique que le processus d'extraction est en cours."
)

st.markdown(
    " - Une fois l'exécution terminée, un bouton apparaîtra pour télécharger le fichier résultant. Veuillez noter que la prédiction peut prendre du temps "
    "en fonction de la taille du fichier. ⏳ Assurez-vous de patienter jusqu'à ce que le processus soit complet avant de télécharger les résultats."
)
st.session_state.pages = None
dt = None
if 'dt' not in st.session_state:
    st.session_state.dt = dt
else:
    dt = st.session_state.dt

# Importation du PDF expander
with st.expander("Importation du PDF.", expanded=True):
    file = st.file_uploader("", type=["pdf"])

    # Check if a new file is uploaded
    if file is not None:
        new_file = True
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf_path = temp_pdf.name  # Save the path before closing

            def show_pdf(file_path):
                with open(file_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf">'
                st.markdown(pdf_display, unsafe_allow_html=True)

            show_pdf(temp_pdf_path)

            # Update session_state
            st.session_state.new_file = new_file
            st.session_state.temp_pdf_path = temp_pdf_path

        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'affichage du PDF : {e}")
    else:
        new_file = False

    # Update session_state
    st.session_state.new_file = new_file

# add button to start PDF analysis
analyze_pdf_button = st.button("Analyser le PDF")

# PDF analysis section
if analyze_pdf_button:
    st.info("L'analyse du PDF est en cours. Veuillez patienter...")

    # Record start time
    start_time = time.time()

    analyzer = dd.get_dd_analyzer()
    # Reset pages and perform PDF analysis
    st.session_state.pages = None          
    df = analyzer.analyze(path=temp_pdf.name)
    df.reset_state()
    
    doc = iter(df)
    pages = []
    for _ in range(len(df)):
        page = next(doc)
        pages.append(page)

    # Record end time
    end_time = time.time()

    st.session_state.pages = pages
    st.success(f"L'analyse du PDF est terminée. Temps d'exécution : {end_time - start_time:.2f} secondes")

    # Additional information or actions after PDF analysis
    # You can add more code here if needed.
    if file is not None and st.session_state.new_file and st.session_state.pages is not None:
        st.session_state.pages = pages
        # number of pages
        st.write("Nombre de pages: ", int(len(df)))

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
            return 'Pas Zone'

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

        # Fonction pour traiter
        def process_page(args):
            page_number, (page, next_page) = args
            sorted_layouts = sort_layouts_by_coordinates(page.layouts)

            articles_start_indices = [i for i, layout in enumerate(sorted_layouts) if is_article_start(layout)]
            articles_end_indices = [i - 1 for i in articles_start_indices[1:]] + [len(sorted_layouts) - 1]

            page_data = []
            for start, end in zip(articles_start_indices, articles_end_indices):
                article_layouts = sorted_layouts[start:end + 1]
                article_content = extract_article_content(article_layouts)

                # Check if the article continues on the next page
                if next_page and is_article_end(sorted_layouts[end], next_page.layouts[0]):
                    continuation_layouts = sort_layouts_by_coordinates(next_page.layouts)
                    continuation_content = extract_article_content(continuation_layouts)
                    article_content += continuation_content

                if article_content:
                    page_data.append({'ID': page_number, 'Contenu': article_content,
                                    'Type': predict(classifier_model, article_content)})

            return page_data

        # Usage of the modified process_page function
        pages_data = []
        for page_number, (page, next_page) in enumerate(zip(pages, pages[1:])):
            page_data = process_page((page_number, (page, next_page)))
            pages_data.append(page_data)


        # Concaténer les données de chaque page
        article_data = []
        for page_data in pages_data:
            article_data.extend(page_data)

        count1 = 0
        count3 = 0
        count4 = 0
        count5 = 0

        for page in pages:
            for layout in page.layouts:
                if layout.category_name == "title":
                    count1 += 1
                if layout.category_name == "list":
                    count3 += 1
                if layout.category_name == "figure":
                    count5 += 1
            for table in page.tables:
                count4 += 1
                
        dt = pd.DataFrame(article_data)

        texts = [page.text for page in pages]
        all_text = '\n'.join(texts)
        st.session_state.dt = dt
        st.session_state.new_file = False
        st.session_state.count1=count1
        st.session_state.count3=count3
        st.session_state.count4=count4
        st.session_state.count5=count5
        st.session_state.all_text=all_text

with st.expander("Informations du PDF.", expanded=True):
    if st.session_state.dt is not None:
        dt = st.session_state.dt
        count1=st.session_state.count1
        count3=st.session_state.count1
        count4=st.session_state.count4
        count5=st.session_state.count5
        all_text=st.session_state.all_text
        st.write("Nombre d'articles : ", int(len(dt)))
        st.write("Nombre de titres : ", int(count1))
        st.write("Nombre de listes : ", int(count3))
        st.write("Nombre de figures : ", int(count5))
        st.write("Nombre de tables : ", int(count4))

        st.download_button("⬇️ Télécharger l'intégralité du contenu du PDF en tant que texte", all_text, "text.txt",
                            "text/plain",
                            use_container_width=True)

with st.expander("Affichage des articles et leur classe.", expanded=True):
    if st.session_state.dt is not None:
        dt = st.session_state.dt

        def highlight_greaterthan(row, token_to_highlight=None):
            # iterate on 
            styles = ['background-color: #a24857 ' if token_to_highlight in str(row['Contenu']) else '' for _ in row.index]
            return styles

        dt = st.session_state.dt
        prompt = st.text_input("Entrez un mot à mettre en surbrillance dans les articles.")
        
        if prompt:
            st.dataframe(dt.style.apply(highlight_greaterthan, token_to_highlight=prompt, axis=1))
        else:
            st.dataframe(dt)
        st.download_button("⬇️ Télécharger les articles en tant que .csv", dt.to_csv(), "annotated.csv", use_container_width=True)
    
    # Display the data frame
        # Process each page
        # add a section of articles, table, lists, figures count
# add section displaying infos of figure and table layout

