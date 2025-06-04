
# Subir dos niveles desde src/internals/ hacia la raíz del repo
def load_data():
    from pathlib import Path
    import pandas as pd
    repo_root = Path.cwd().parents[1]

# Construir la ruta al archivo CSV
    ruta_csv = repo_root / "Data" / "Data.csv"

# Verificar si el archivo existe antes de cargarlo
    if ruta_csv.exists():
        df = pd.read_csv(ruta_csv, sep=None, engine="python")
        return df
    else:
        print("❌ El archivo no se encontró en esa ruta.")