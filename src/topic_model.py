from umap import UMAP
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class TopicModeler:
    """
        Clase que identifica los tópicos de un tema. Se tienen los miembros de
        docs:Dataframe pandas, los conjunto de comentarios a revisar
        topics: lista de enteros, en la que identifica los tópics
        probs:numpy array, probabilidades de frecuencia que existen en los tópicos
        freq_info: dataframe de pandas, la información completa por tópico
        topic_titles: arreglo de string, ,titulos de los tópicos
        Se tienen un método constructor que permite inicializar los atributos en None, además de pasarle los páraemtros para poder
        ajustar el modelo de berTOPIC, así como un modelo UMAP por defecto
        La función fit, que entrena el modelo
        _extract_titles, que extrae las uniones de palabras de los tópicos encontrados
        Métodos de visualización como _visualize_chart, que permite desplegar en forma de 
        gráfica de barras los tópicos encontrados
        visualize_hierarchy: permite ver de forma jerarquizadas los orígenes de algunos tópicos y como se fueron
        "independizando"
        visualize_topics: refleja los tópicos encontrados en un cierto conjunto en un pano cartesiano
        plot_wordcloud: muestra nubes de palabras de un tópico
    """
    def __init__(self,topic_vectorizer:TfidfVectorizer,language="spanish",params:dict=None):
        """Método constructor que inicializa los miembros y párametros de configuración para berTOPIC
        Args:
            topic_vectorizer (TfidfVectorizer): Vector de frecuencias calculado y generado de un corpus de palabras
            language (str, optional): Idioma en el que se desea hacer el análisis. Defaults to "spanish".
            params (dict, optional):Diccionario en dónde se le envían los párametros de entrenamiento
            a berTOPIC. Se agregan todos aquellos que pueda recibir dicho modelo. Defaults to None.
        """
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
    def fit(self,docs:pd.DataFrame)->tuple[list[int],np.ndarray|None]:
        """Función que entrena el modelo
        Args:
            docs (pd.DataFrame): Dataframe que posee los tópicos a entrenar
        Returns:
            tuple[list[int],np.ndarray|None]: Tupla con la lista de número asignado a tópico y las probabilidades o frecuencias
            de tópicos
        """
        self.docs=docs
        self.topics,self.probs=self.model.fit_transform(docs)
        self.freq_info=self.model.get_topic_info()
        self.topic_titles=self._extract_titles(self.freq_info)
        return  self.topics,self.probs

    def _extract_tiles(self,freq:pd.DataFrame)->list[str]:
        """Extrae los título de los tópicos extraídos, según a las palabras que haya encontrado
        Args:
            freq (pd.DataFrame): Posee las representaciones, número de tópico y comentarios releventaes para determinar el
            tópico 

        Returns:
            list[str]: Retorna la lista de tópicos unidos por palabras encontradas en la Representation del conjunto de datos.

        """
        array=[]
        for _,row in freq.iterrows():
            title=" ".join(row["Representation"])
            array.append(title)
        return array
    def freq(self)->list[str]|None:
        """Función getter que obtiene los títulos generados

        Returns:
            list[str]|None: Lista de titulos generados por la función _extract_titles
        """
        return self.topic_titles
    ###Visualizaciones #####
    def visualize_barchart(self, top_n_topics=10, n_words=10):
        """Gráfico de barras de palabras más representativas
        Args:
            top_n_topics (int, optional): Número de tópicos a buscar principalmente. Defaults to 10.
            n_words (int, optional): Número de palabras límite por tópicos. Defaults to 10.

        Returns:
            any: Genera un gráfico de barras
        """
        return self.model.visualize_barchart(top_n_topics=top_n_topics, n_words=n_words)

    def visualize_hierarchy(self):
        """Jerarquía de tópicos

        Returns:
            Figure: Genera el gráfico de jerarquía
        """
        return self.model.visualize_hierarchy()

    def visualize_topics(self):
        """Mapa 2D de los tópicos
        Returns:
            Figure: Genera el gráfico en un plano cartesiano viendo los agrupamientos de tópicos
        """
        return self.model.visualize_topics()
    def plot_wordcloud(self,topic_id=None):
        """Genera una nube de palabras

        Args:
            topic_id (entero, optional): _description_. id de tópico a graficar
        """
        if topic_id is None:
            words=self.model.get_topics()
            text=" ".join([w for tfid in words for w,_ in words[tfid]])
        else:
            words=dict(self.model.get_topic(topic_id))
            text=" ".join([w for w in words.keys()])
        wc=WordCloud(width=800,height=400,background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nube de palabras del tópico {topic_id}" if topic_id else "Nube global")
        plt.show()