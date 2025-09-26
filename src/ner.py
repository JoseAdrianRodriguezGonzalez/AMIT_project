import spacy

class NERextractor:
    def __init__(self, model_name="es_core_news_md", stop_words=None):
        self.nlp = spacy.load(model_name)
        self.stop_words = set(stop_words) if stop_words else set()
        self.target_entities = {"LOC", "ORG", "PER"}  # opcional: filtrar por tipo
        self.docs = None
        self.entities = None
        self.top_entities_list = None

    def extract_entities(self, docs, filter_types=True):
        """
        Devuelve lista de entidades por documento
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

    def top_entities(self, n=10):
        """
        Devuelve Top n entidades globales, filtrando stopwords
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
    def get_entities(self):
        return self.entities

    def get_top_entities(self):
        return self.top_entities_list
