import nltk
from nltk.corpus import stopwords
class Preprocessor:
    """Preprocessor
    Clase que hace un preprocesamiento ligero de texto
    Posee el atributo stopword, que indica la lista de stopswords en un idioma determinado
    Posee su método constructor y una función que formatea al conjunto de datos
    """
    def __init__(self,language="spanish"):
        """Método constructor que inicializa el atributo de stepwords
        Args:
            language (str, optional): Cadena que indica en que idioma se necesitan los stepwords
            . Defaults to "spanish".
        """
        nltk.download("stopwords",quiet=True)
        self.stopword = stopwords.words(language)
    def transform(self,df):
        """Función que se dedica a transformar las columnas de los comentarios si es necesario

        Args:
            df (pd.Dataframe): Es un conjunto de datos de pandas

        Raises:
            ValueError: Si el formato no es como el esperado

        Returns:
            lista: Los comentarios en forma de lista 
        """
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