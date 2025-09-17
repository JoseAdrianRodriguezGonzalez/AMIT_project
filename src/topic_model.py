from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
class TopicModeler:
    def __init__(self,topic_vectorizer,language="spanish",params=None):
        if params is None:
            params={}
        if "umap_model" not in params:
            params["umap_model"] = UMAP(n_neighbors=2, n_components=2, metric='cosine', random_state=42)

        self.model=BERTopic(language=language,
                            vectorizer_model=topic_vectorizer,calculate_probabilities=True,**params)
    def fit(self,docs):
        return  self.model.fit_transform(docs)
    def freq(self):
        return self.model.get_topic_info()
    
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