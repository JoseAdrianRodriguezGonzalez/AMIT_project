import nltk
from nltk.corpus import stopwords
class Preprocessor:
    def __init__(self,language="spanish"):
        nltk.download("stopwords",quiet=True)
        self.stopword = stopwords.words(language)
    def transform(self,df):
        if "Titulo" in df.columns and "Review" in df.columns:
            docs=(
                df["Titulo"].fillna('') + " " +
                df["Review"].fillna('')
            ).tolist()
        elif "Comentarios" in df.columns:
            docs=df["Comentarios"].fillna("").tolist()
        else:
            raise ValueError("El DataFrame no tiene las columnas esperadas (Titulo+Review o Comentarios).")
        return docs