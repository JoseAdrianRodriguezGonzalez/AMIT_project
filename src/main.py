from data_loader import data_loader,os
from preproccessing import Preprocessor
from vectorizer import Vectorizer
from topic_model import TopicModeler
from ner import NERextractor
def main():
    #Loading data
    lista=["../data/"+f for f in os.listdir("../data") if f.endswith("csv")]
    loader=data_loader(lista)
    datasets=loader.load()
    #dataset traversal
    for name,df in datasets.items():
        #Initializing the elements
        pre=Preprocessor()
        vec=Vectorizer(pre.stopword)

        print(f"\n{'='*50}")
        print(f"procesando {name}")
        print(f"\n{'='*50}")
        docs=pre.transform(df)
        print(f"se generaron {len(docs)} documentos")

        ner=NERextractor(stop_words=pre.stopword)
        entities=ner.extract_entities(docs,filter_types=False)
        top_10=ner.top_entities(entities=entities)
        print(f"Top 10 entidades para {name}:")
        for ent, count in top_10:
            print(f"  {ent}: {count}")
        # Filtra los stepwords
        X,vocab=vec.build_vocab(docs)
        print(f"TF-IDF con {len(vocab)} palabras únicas")
        print("Ejemplo de vocabulario:", vocab[:20])  # mostrar las primeras 20 palabra
        #Vector definitivo
        vector=vec.get_topic_vectorizer()
        tp=TopicModeler(vector)
        topics,probs=tp.fit(docs)
        freq=tp.freq()
        print(f"Topics {topics}")
        print(f"Probabilidades {probs}")
        print(f"frecuencias {freq}")
        fig1 = tp.visualize_barchart(top_n_topics=10, n_words=10)
        fig3 = tp.visualize_hierarchy()
        fig2 = tp.visualize_topics()

        # 4. Mostrar (plotly)
        fig1.show()
        fig2.show()
        fig3.show()
if __name__=="__main__":
    main()