from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, stopwords, max_df=0.3):
        self.stopwords = stopwords
        self.max_df = max_df
        self.global_vectorizer = None
        self.filtered_vocab = None
        self.topic_vectorizer = None

    def build_vocab(self, docs):
        """Primera pasada: crea vocabulario filtrado"""
        self.global_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            max_df=self.max_df
        )
        X = self.global_vectorizer.fit_transform(docs)
        self.filtered_vocab = self.global_vectorizer.get_feature_names_out()
        return X, self.filtered_vocab

    def get_topic_vectorizer(self):
        """Crea vectorizer definitivo con vocab filtrado"""
        if self.filtered_vocab is None:
            raise ValueError("Debes ejecutar build_vocab primero")
        self.topic_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            vocabulary=self.filtered_vocab
        )
        return self.topic_vectorizer
