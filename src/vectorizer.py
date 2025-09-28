from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    """
    Clase que vectoriza las palabras previamente filtradas en el proceso de preprocessing.
    Se tienen los atributos de stepwords, para identificar aquellas palabras que son muy frecuentes por tópico.
    max_df, que es la máxima tolerancia aceptada para tolerar que es o no un stepword adicional
    global_vectorizer, donde se guarda el vector resultante
    filtered_vocab, es el corpus formado después de vectorizar y filtrar palabras
    topic_vectorizer, segundo proceso de filtrado 
    Posee el método constructor, el método de build vocab, que permite construir el vocabulario y la segunda
    revisada de vectorización
    """
    def __init__(self, stopwords:list[str], max_df=0.3):
        """Método constructor que permite inicializar cada una de los atributos

        Args:
            stopwords (list): Es una lista de strings que tiene los stepwords
            max_df (float, optional): Cantidad máxima de detección de stepword o palabra demasiado frecuente
            bajo el contexto del tópico y que no aporte información relevante. Defaults to 0.3.
        """
        self.stopwords = stopwords
        self.max_df = max_df
        self.global_vectorizer = None
        self.filtered_vocab = None
        self.topic_vectorizer = None

    def build_vocab(self, docs):
        """Primera pasada: crea vocabulario filtrado através de los step words de NLTK
        Args:
            docs (pd.Dataframe): tópicos después de un acómodo de datos por parte del conjunto de datos 

        Returns:
            tupla de sparse matrix y numpy array: Retorna la matriz generada después de calcular su frecuencia de TFID
            y el numpy array contiene las palabras encontradas 
        """
    
        self.global_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            max_df=self.max_df
        )
        X = self.global_vectorizer.fit_transform(docs)
        self.filtered_vocab = self.global_vectorizer.get_feature_names_out()
        return X, self.filtered_vocab

    def get_topic_vectorizer(self):
        """Crea vectorizer definitivo con vocab filtrado si en dado caso existen palabras irrelevantes
        Raises:
            ValueError:En caso de no haber ejecutado el primer proceso de vectorización
        Returns:
            matriz de tfid: matriz de tfid caracteristicas
        """
        if self.filtered_vocab is None:
            raise ValueError("Debes ejecutar build_vocab primero")
        self.topic_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            vocabulary=self.filtered_vocab
        )
        return self.topic_vectorizer
