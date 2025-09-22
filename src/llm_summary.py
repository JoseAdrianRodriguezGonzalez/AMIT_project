from llama_cpp import Llama
import re
class LlamaModel:
    def __init__(self,model):
        self.model_name=model
        self.model=Llama(model_path=model,
                             n_ctx=4096,          # contexto (ajusta según el modelo)
                            n_gpu_layers=-1,     # -1 = cargar todas las capas en GPU
                            n_threads=8,
                                      verbose=False          )
    def request_title(self,freq,params=None):
        if params==None:
            params = {
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": ["\n\n"]  # por ejemplo, para cortar al final de un párrafo
            }

        titles=[]
        for t in freq:
            prompt = f"""
            Eres un redactor creativo de turismo y viajes en español.
            Debes escribir un único título atractivo y natural para un artículo promocional de viaje.

            Condiciones:
            - Usa estas palabras clave de manera implícita:
            {t}
            - No hagas preguntas, listas, ni devuelvas explicaciones.
            - No devuelvas diccionarios ni estructuras raras.
            - El título debe sonar como encabezado de revista de turismo.
            - Devuelve solo la frase, sin comillas ni signos de interrogación.

            Título turístico:
            """
            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            print(output)
            titles.append(out)
        return titles
    def request_interpretation(self,freq,params=None):
        if params==None:
            params = {
                "max_tokens": 400,
                "temperature": 0.4,
                "top_p": 0.9,
                "repeat_penalty": 1.5,
                "stop": ["\n\n"]  # por ejemplo, para cortar al final de un párrafo
            }

        titles=[]
        for t in freq:
            prompt = f"""
            Eres un experto en turismo que escribe en español.
            Debes redactar un párrafo breve (2–3 frases) para una guía de viajes.

            Condiciones:
            - Utiliza las siguientes palabras clave de manera implícita, conectándolas con lógica natural:
            {t}
            - No inventes lugares que no estén en las palabras clave.
            - No hagas preguntas, ni listas, ni devuelvas explicaciones.
            - Usa un tono inspirador y turístico.
            - Devuelve únicamente el párrafo, sin comillas ni código.

            Descripción turística:
            """
            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            titles.append(out)
        return titles