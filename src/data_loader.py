import pandas as pd
import os
class data_loader:
    """Esta clase consiste en poder cargas diferentes tipos de archivos.
        Como atributos tiene una lista de nombers de archivos que puedan estar contenidos dentro de una carpeta
        Posee los método de cargar archivo
    """
    def __init__(self,files:str|iter[str]):
        """_summary_
        Método constructor que almacena el nombre o los nombres de los archvios
        Args:
            files (lista o string): Nombre o nombres de archivos

        Raises:
            ValueError: Erro en dado caso de que el archivo se encontró que no tiene una extensión
            FileNotFoundError: Erro en dado caso que el archivo con determinado nombre no exista
        """
        if isinstance(files,str):
            files=[files]
        self.files=files
        for name_file in files:
            if "." not in name_file:
                raise ValueError("El archivo debe tener una extensión (ejemplo: data.csv)")
            if not os.path.exists(name_file):
                raise FileNotFoundError(f"El archivo {name_file} no existe")
    def load(self):
        """_summary_
        Carga los datos del (los) archivo(s)en forma de diccionario.
        Las keys son el nombre del archivo y el value es su contenido 
        Raises:
            ValueError: Si el formato que recupero no está dentro de los válidos
            RuntimeError: Si al cargar los datos de los elementos encontrados no se encuentre completo o tenga
            errores de lectura

        Returns:
            dict: Diccionario que contiene los datasets.
        """
        datasets={}
        for f in self.files:
            try:
                if f.endswith(".csv"):
                    datasets[f]= pd.read_csv(f)
                elif f.endswith((".xls",".xlsx")):
                     datasets[f]=pd.read_excel(f)
                else:
                    raise ValueError(f"formato de archivo de {f} no soportado")
            except Exception as e:
                raise RuntimeError(f"Error al cargar {f}: {e}")
        return datasets