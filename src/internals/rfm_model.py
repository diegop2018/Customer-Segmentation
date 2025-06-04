import pathlib as Path
import pandas as pd

# Subir dos niveles desde src/internals/ hacia la raíz del repo
from pathlib import Path
import pandas as pd

repo_root = Path.cwd().parents[2]

# Construir la ruta al archivo CSV
ruta_csv = repo_root / "Data" / "Data.csv"

# Verificar si el archivo existe antes de cargarlo
df = pd.read_csv(ruta_csv, sep=None, engine="python")


# Mostrar las primeras filas del DataFrame
print(df.head(3))

def __init__(self, data_path=None, df=None):
        """
        Inicializar con datos desde archivo CSV o DataFrame
        """
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path, sep=None, engine="python")
        else:
            # Cargar datos por defecto
            repo_root = Path.cwd().parents[1]
            ruta_csv = repo_root / "Data" / "Data.csv"
            self.df = pd.read_csv(ruta_csv, sep=None, engine="python")
        
        self.df_rfm = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.results_comparison = []