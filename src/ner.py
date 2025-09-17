import spacy

class NERextractor:
    def __init__(self, model_name="es_core_news_md", stop_words=None):
        self.nlp = spacy.load(model_name)
        self.stop_words = set(stop_words) if stop_words else set()
        self.target_entities = {"LOC", "ORG", "PER"}  # opcional: filtrar por tipo

    def extract_entities(self, docs, filter_types=True):
        """
        Devuelve lista de entidades por documento
        """
        all_entities = []
        for doc in self.nlp.pipe(docs, batch_size=32):
            if filter_types:
                ents = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in self.target_entities]
            else:
                ents = [(ent.text, ent.label_) for ent in doc.ents]
            all_entities.append(ents)
        return all_entities

    def top_entities(self, entities, n=10):
        """
        Devuelve Top n entidades globales, filtrando stopwords
        """
        counts = {}
        for doc in entities:
            for text, label in doc:
                key = text.lower().strip()
                if key in self.stop_words or len(key) < 2:
                    continue
                counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
