import spacy
import pandas as pd
nlp=spacy.load("es_core_news_md")

df=pd.read_csv("data/acuario.csv")

docs = (df['Titulo'].fillna('') + " " + df['Review'].fillna('')).tolist()

all_entities=[]
for doc in nlp.pipe(docs,batch_size=32):
    ents=[(ent.text,ent.label_) for ent in doc.ents]
    all_entities.append(ents)
df["Entidades"] =all_entities
df.to_csv("./data/acuario_con_ner.csv", index=False)
print(df[["Titulo","Entidades"]].head(20))