import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
import matplotlib.pyplot as plt

# ====================
# 1️⃣ Stopwords en español
# ====================
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# ====================
# 2️⃣ Cargar dataset
# ====================
df = pd.read_csv('./data/acuario.csv')
df = df.drop(columns=['TipoViaje','OrigenAutor'], errors='ignore')
docs = (df['Titulo'].fillna('') + " " + df['Review'].fillna('')).tolist()

# ====================
# 3️⃣ TF-IDF global para filtrar palabras muy frecuentes
# ====================
tfidf_global = TfidfVectorizer(
    stop_words=spanish_stopwords,
    max_df=0.5,  # eliminar palabras que aparecen en más del 50% de los documentos
    min_df=2     # eliminar palabras que aparecen menos de 2 veces
)
X_tfidf = tfidf_global.fit_transform(docs)
filtered_vocab = tfidf_global.get_feature_names_out()

# ====================
# 4️⃣ Crear BERTopic con vocabulario filtrado
# ====================
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

# ====================
# 5️⃣ Revisar tópicos más comunes
# ====================
freq = topic_model.get_topic_info()
print(freq[['Topic','Count','Representation']].head(10))

# ====================
# 6️⃣ Visualizaciones
# ====================

# 6a. Gráfico de barras con top palabras por tópico
fig1 = topic_model.visualize_barchart(top_n_topics=10, n_words=10)
fig1.show()

# 6b. Mapa de tópicos (proyección 2D)
fig2 = topic_model.visualize_topics()
fig2.show()

# 6c. Jerarquía de tópicos
fig3 = topic_model.visualize_hierarchy()
fig3.show()
