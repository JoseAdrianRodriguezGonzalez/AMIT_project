import os
import pandas as pd
files=os.listdir("./data")
files
df=pd.read_csv('./data/acuario.csv')
df[['Titulo','Review','TipoViaje','Calificacion','OrigenAutor','FechaOpinion','FechaEstadia']].isna().sum()
df.drop(columns=['TipoViaje','OrigenAutor'])
docs=(df['Titulo'].fillna('')+" "+ df['Review'].fillna('')).tolist()
from bertopic import BERTopic
topic_model=BERTopic(language='spanish')
topics,probs=topic_model.fit_transform(docs)