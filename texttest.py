import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

# Cargar dataset con entidades
df = pd.read_csv("./data/acuario_con_ner.csv")

docs = (df["Titulo"].fillna("") + " " + df["Review"].fillna("")).tolist()

vectorizer_model = TfidfVectorizer(stop_words=spanish_stopwords, max_df=0.3)
topic_model = BERTopic(language="spanish", vectorizer_model=vectorizer_model)

topics, probs = topic_model.fit_transform(docs)

freq = topic_model.get_topic_info()
print(freq.head(15))
