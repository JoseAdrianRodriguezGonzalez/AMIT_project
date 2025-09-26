from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
class TopicModeler:
    def __init__(self,topic_vectorizer,language="spanish",params=None):
        if params is None:
            params={}
        if "umap_model" not in params:
            params["umap_model"] = UMAP(n_neighbors=2, n_components=2, metric='cosine', random_state=42)

        self.model=BERTopic(language=language,
                            vectorizer_model=topic_vectorizer,calculate_probabilities=True,**params)
        self.docs = None
        self.topics = None
        self.probs = None
        self.freq_info = None
        self.topic_titles = None
    def fit(self,docs):
        self.docs=docs
        self.topics,self.probs=self.model.fit_transform(docs)
        self.freq_info=self.model.get_topic_info()
        self.topic_titles=self._extract_titles(self.freq_info)
        return  self.topics,self.probs

    def _extract_tiles(self,freq:pd.Dataframe):
        array=[]
        for _,row in freq.iterrows():
            title=" ".join(row["Representation"])
            array.append(title)
        return array
    def freq(self):
        return self.topic_titles
    ###Visualizaciones #####
    def visualize_barchart(self, top_n_topics=10, n_words=10):
        """Gráfico de barras de palabras más representativas"""
        return self.model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words)

    def visualize_hierarchy(self):
        """Jerarquía de tópicos"""
        return self.model.visualize_hierarchy()

    def visualize_topics(self):
        """Mapa 2D de los tópicos"""
        return self.model.visualize_topics()