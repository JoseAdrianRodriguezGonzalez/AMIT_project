import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
class TopicNERAnalyzer:
    """
    Clase integradora que combina análisis de tópicos (BERTopic) con extracción de entidades (NER).
    """

    def __init__(self, topic_modeler: 'TopicModeler', ner_extractor: 'NERextractor'):
        self.topic_modeler = topic_modeler
        self.ner_extractor = ner_extractor
        self.topic_entities = None  # dict[int, list[tuple[str, str]]]
        self.entity_summary = None  # pd.DataFrame resumen global

    def fit(self, docs: pd.DataFrame):
        """
        Entrena el modelo de tópicos y luego asocia entidades a cada tópico.
        """
        topics, probs = self.topic_modeler.fit(docs)
        entities = self.ner_extractor.extract_entities(docs)
        topic_entities = {}
        for topic_id, ents in zip(topics, entities):
            if topic_id not in topic_entities:
                topic_entities[topic_id] = []
            topic_entities[topic_id].extend(ents)
        self.topic_entities = topic_entities

        #  resumen global por tópico
        summary = []
        for topic_id, ents in topic_entities.items():
            counter = Counter([text.lower() for text, label in ents])
            most_common = counter.most_common(5)
            summary.append({
                "topic_id": topic_id,
                "topic_title": self.topic_modeler.topic_titles[topic_id] if topic_id < len(self.topic_modeler.topic_titles) else "N/A",
                "entities": most_common
            })
        self.entity_summary = pd.DataFrame(summary)
        return self.entity_summary

    def get_entities_by_topic(self, topic_id: int) -> list[tuple[str, str]]:
        """Devuelve las entidades asociadas a un tópico específico."""
        if self.topic_entities is None:
            raise ValueError("Debes ejecutar fit() antes.")
        return self.topic_entities.get(topic_id, [])

    def visualize_topic_entities(self, topic_id: int, n=10):
        """Grafica las entidades más frecuentes por tópico."""
        if self.topic_entities is None:
            raise ValueError("Ejecuta fit() primero.")


        ents = [text for text, _ in self.topic_entities.get(topic_id, [])]
        if not ents:
            print(f"No hay entidades para el tópico {topic_id}")
            return

        text = " ".join(ents)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Entidades más frecuentes en el tópico {topic_id}")
        plt.show()


ASPECT_PROTOTYPES = {
    "PRICE": [
        "precio elevado",
        "entrada costosa",
        "demasiado caro",
        "no vale lo que cuesta"
    ],
    "CROWD": [
        "demasiada gente",
        "muy lleno",
        "largas filas",
        "aforo saturado"
    ],
    "ANIMALS": [
        "animales",
        "especies",
        "bienestar animal",
        "pingüinos",
        "tiburones"
    ],
    "FOOD": [
        "comida",
        "restaurante",
        "cafetería",
        "alimentos"
    ],
    "SERVICE": [
        "atención del personal",
        "mal servicio",
        "trato del staff"
    ],
    "FACILITY": [
        "instalaciones",
        "baños",
        "infraestructura",
        "espacios"
    ],
    "ACTIVITY": [
        "actividades",
        "interacción",
        "alimentar animales",
        "juegos"
    ],
    "LOCATION": [
        "zona",
        "túnel",
        "nivel",
        "área específica"
    ]
}

def classify_aspect_semantic(text,embedder):
    text_vec =embedder.encode(text,normalize_embeddings=True)
    best_label=None
    best_score=0.0
    for label, prototypes in  ASPECT_PROTOTYPES.items():
        proto_vecs= embedder.encode(prototypes,normalize_embeddings=True)
        score= max(text_vec @ proto_vecs.T)
        if score>best_score:
            best_label=label
            best_score=score
    return best_label,best_score
