import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
import matplotlib.pyplot as plt
#Stop words
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")
#Cargar dataset
df = pd.read_csv('./data/chapultepec.csv')
df = df.drop(columns=['TipoViaje','OrigenAutor'], errors='ignore')
docs = (df['Titulo'].fillna('') + " " + df['Review'].fillna('')).tolist()
#TF-IDF
tfidf_global = TfidfVectorizer(
    stop_words=spanish_stopwords,
    max_df=0.3  
)
X_tfidf = tfidf_global.fit_transform(docs)
filtered_vocab = tfidf_global.get_feature_names_out()
# BERTopic
vectorizer_model = TfidfVectorizer(
    stop_words=spanish_stopwords,
    vocabulary=filtered_vocab
)

topic_model = BERTopic(
    language='spanish',
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(docs)
#topicos comunes
freq = topic_model.get_topic_info()
print(freq[['Topic','Count','Representation']].head(15))

# 6a. Gráfico de barras con top palabras por tópico
fig1 = topic_model.visualize_barchart(top_n_topics=10, n_words=10)
fig1.show()
# 6c. Jerarquía de tópicos
fig3 = topic_model.visualize_hierarchy()
fig3.show()

# 6b. Mapa de tópicos (proyección 2D)
fig2 = topic_model.visualize_topics()
fig2.show()

