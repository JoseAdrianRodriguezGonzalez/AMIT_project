import spacy
import pandas as pd
class NERextractor:
    """Clase que posee el extractor de entidades representativas
    Posee los siguientes atributos:
    nlp: contiene el corpus con el que se entrena ner
    stop_words: las palabras que no aportan información relevante
    target_entities: Entidades a buscar
    entities: entidades encontradas
    top_entities_list: Lista de los top 10 de entidades que existen 
       """
    def __init__(self, model_name="es_core_news_md", stop_words=None):
        self.nlp = spacy.load(model_name)
        self.stop_words = set(stop_words) if stop_words else set()
        self.target_entities = {"LOC", "ORG", "PER"}  # opcional: filtrar por tipo
        self.docs = None
        self.entities = None
        self.top_entities_list = None

    def extract_entities(self, docs:pd.DataFrame, filter_types:bool=True)->list[list[tuple[str,str]]]:
        """        Devuelve lista de entidades por documento

        Args:
            docs (pd.DataFrame): Dataframe de los comentarios
            filter_types (bool, optional): Si se tiene o requiere aplica un filtro exhaustivo. Defaults to True.
        Returns:
            list[list[tuple[str,str]]]: Contiene las entidades por documento
        """
        all_entities = []
        self.docs=docs
        for doc in self.nlp.pipe(docs, batch_size=32):
            if filter_types:
                ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in self.target_entities]
            else:
                ents = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities.append(ents)
        self.entities=all_entities
        return all_entities

    def top_entities(self, n:int=10)->list[tuple[str,int]]:
        """Devuelve Top n entidades globales, filtrando stopwords

        Args:
            n (int, optional): número de top de entidades. Defaults to 10.

        Raises:
            ValueError: Se ejecuta si no se ha extraído una entidad

        Returns:
            list[tuple[str,int]]: Lista de tuplas que viene el orden y la cantidad de veces que es popular esa entidad
        """
        if self.entities is None:
            raise ValueError("Primero ejecuta extract")
        counts = {}
        for doc in self.entities:
            for text, label in doc:
                key = text.lower().strip()
                if key in self.stop_words or len(key) < 2:
                    continue
                counts[key] = counts.get(key, 0) + 1
        self.top_entities_list=sorted(counts.items(),key=lambda x:x[1],reverse=True)[:n]
        return self.top_entities
    def get_entities(self)->list|None:
        """Obtiene las entidades encontrados

        Returns:
            list|None: Lista de entidades
        """
        return self.entities

    def get_top_entities(self)->list[tuple]|None:
        """Devuelve la lista de entidades

        Returns:
            list[tuple]|None: Lista de tuplas de popularidad de tópicos de entidades
        """
        return self.top_entities_list
