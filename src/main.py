from data_loader import data_loader,os
from preproccessing import Preprocessor
from vectorizer import Vectorizer
from topic_model import TopicModeler
from ner import NERextractor
from llm_summary import LlamaModel
def main():
    #Loading data
    lista=["../data/"+f for f in os.listdir("../data") if f.endswith("csv")]
    chol="../data/comentarios_cholula.xlsx"
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
        title=tp.titles(freq)
      #  for t in title:
      #      print(t)
      #  llama=LlamaModel("../model/qwen2.5-3b-instruct-q4_k_m.gguf")
      #  descriptive_titles=llama.request_title(title)
      #  for t in descriptive_titles:
      #      print(t)
     #   descriptive_inter=llama.request_interpretation(title)
      #  for t in descriptive_inter:
      #      print(t)
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