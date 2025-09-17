import pandas as pd
import os
class data_loader:
    def __init__(self,files):
        if isinstance(files,str):
            files=[files]
        self.files=files
        for name_file in files:
            if "." not in name_file:
                raise ValueError("El archivo debe tener una extensión (ejemplo: data.csv)")
            if not os.path.exists(name_file):
                raise FileNotFoundError(f"El archivo {name_file} no existe")
    def load(self):
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