from llama_cpp import Llama
import re
import sys, os
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
class LlamaModel:
    """Clase que posee la iteracción con un modelo de LLM. Este modelo puede ser llama u otro tipo de modelo de GGUF
    model_name=nombre del modelo
    model=instancia de llama que inicia el modelo
    freq: las palabras que se usarán para dar un inferencia
    generetaed_titles : titutlos generados 
    generated_interpretations: Interpretaciones generadas 
    """
    def __init__(self,model:str):
        """Método constructor que inicializa todos los atributos
        Args:
            model (str): Nombre del modelo
        """
        self.model_name=model
        with suppress_stderr():
            self.model=Llama(model_path=model,
                                n_ctx=4096,          
                                n_gpu_layers=32,  
                                n_threads=8,
                                verbose=False)
        self.freq = None
        self.generated_titles = None
        self.generated_interpretations = None

    def request_title(self,freq:list[list[str]],params:dict=None)->list[str]:
        """Se crea/genera un titulo dado a una lista de tópics

        Args:
            freq (list[list[str]]): lista de tópicos de palabras que provienen de tópicos extraídos por berTOPIC
            params (dict, optional): Diccionario de parámetros que se le pueden pasar al modelo
            con el fin de mejorar o cambiar la inferencia que resulta. Defaults to None.

        Returns:
            list[str]: lista de titulos que salieron por el conjunto de tópicos
        """        
        if params==None:
            params = {
                "max_tokens": 100,
                "temperature": 0.4,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": ["\n\n"]  # por ejemplo, para cortar al final de un párrafo
            }
        self.freq=freq
        titles=[]
        for t in freq:
            prompt = f"""
            Eres un redactor creativo de turismo y viajes en español Mexicano.
            Debes escribir un único título atractivo y natural para un artículo promocional de viaje.

            Condiciones:
            - Usa estas palabras clave de manera implícita, provienen de un análisis de berTOPIC:
            {t}
            - No hagas preguntas, listas, ni devuelvas explicaciones.
            - No devuelvas diccionarios ni estructuras raras.
            - El título debe sonar como encabezado de revista de turismo.
            - Devuelve solo la frase, sin comillas ni signos de interrogación.
            - No divages, ni asumas con lugares que no se proporcionan en los comentarios/tópicos que te estoy pasando.
            - Si te dicen explícitamente un lugar, mencionalo.
            
            Título turístico:
            """
            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            out = out.replace("\\n", " ").replace("\n", " ").strip()
            out = out.replace("'", "").replace('"', "")
            out = re.sub(r'#.*', '', out)         
            out = re.sub(r'return.*', '', out)    
            out = re.sub(r'articles\s*=.*', '', out)  
            out = re.sub(r'\s{2,}', ' ', out).strip()
            if len(out.split()) > 3:  
                titles.append(out)
        print(len(titles))
        self.generated_titles=titles
        return titles
    def request_interpretation(self,freq:list[list[str]],params:dict=None)->list[str]:
        """En base a un conjunto de palabras de tópicos, genera interpretaciones por tópicos
        Args:
            freq (list[list[str]]):Se tiene la lista de tópicos y dentro las palabras de donde proviene
            params (dict, optional): Diccionario de parámetros para modificar el como funciona el 
            modelo de lenguaje. Defaults to None.

        Returns:
            list[str]: Conjunto de descripciones por tópico
        """        
        if params==None:
            params = {
                "max_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.5,
                "stop": ["\n\n"]  # por ejemplo, para cortar al final de un párrafo
            }
        self.freq=freq
        titles=[]
        for t in freq:
            prompt = f"""
            Eres un experto en turismo que escribe en español.
            Debes redactar un párrafo breve (2–3 frases) para una guía de viajes.

            Condiciones:
            - Utiliza las siguientes palabras clave de manera implícita, que provienen de un análisis de tópicos. Conectalas con lógica natural:
            {t}
            - No inventes lugares que no estén en las palabras clave.
            - No hagas preguntas, ni listas, ni devuelvas explicaciones.
            - Usa un tono inspirador y turístico.
            - Devuelve únicamente el párrafo, sin comillas ni código.
            - Si hay un lugar, retente al contexto del texto, no menciones lugares que no se mencionen en el texto
            Descripción turística:
            """
            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            out = out.replace("\\n", " ").replace("\n", " ").strip()
            out = out.replace("'", "").replace('"', "")
            out = re.sub(r'#.*', '', out)         
            out = re.sub(r'return.*', '', out)    
            out = re.sub(r'articles\s*=.*', '', out)  
            out = re.sub(r'\s{2,}', ' ', out).strip()
            if len(out.split()) > 3:  
                titles.append(out)
            print(len(titles))  
        self.generated_interpretations=titles
        return titles
    def get_titles(self)->list[str]:
        """función que obtiene el conjunto de titulos

        Returns:
            list[str]: conjunto de strings con los títulos generados
        """        
        return self.generated_titles

    def get_interpretations(self)->list[str]:
        """Obtiene las interpretaciones almacenadas

        Returns:
            list[str]: conjunto de strings con las interpretaciones por tópico
        """
        return self.generated_interpretations