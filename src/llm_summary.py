from llama_cpp import Llama
import re
class LlamaModel:
    def __init__(self,model):
        self.model_name=model
        self.model=Llama(model_path=model)
    def request_title(self,freq,params=None):
        if params==None:
            params = {
                "max_tokens": 0,
                "temperature": 0.9,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": ["<|user|>", "<|system|>"]
            }

        titles=[]
        for t in freq:
            print(t)
            prompt = f"""<|system|>Eres un asistente de turismo experto en español.
            Genera un **único título breve, claro, atractivo y coherente** basado en las palabras que te doy.
            Debe leerse como un título de revista o blog turístico, sin signos de interrogación y sin repetir títulos anteriores.
            Prioriza que sea natural y atractivo, puedes modificar las palabras,a manera que quede un texto lo más natural y humano, no solo una lista de palabras.<|end|>
            <|user|>Palabras clave: {t}<|end|>
            <|assistant|>"""
    

            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            print(out)
            titles.append(out)
        return titles
    def request_interpretation(self,freq,params=None):
        if params==None:
            params = {
                "max_tokens": 500,
                "temperature": 0.9,
                "top_p": 0.5,
                "repeat_penalty": 1.5,
                "stop": ["<|user|>", "<|system|>"]
            }

        titles=[]
        for t in freq:
            print(t)
            prompt = f"""Eres un asistente experto en turismo en español. 
            Toma las siguientes palabras clave y genera una descripción coherente, atractiva y completa, de 2-3 frases, sobre el lugar o experiencia. No hagas preguntas ni repitas palabras innecesarias.
            Palabras clave: {t}
            """


            output=self.model(prompt=prompt,**params)
            out=output["choices"][0]["text"]
            print(out)
            titles.append(out)
        return titles