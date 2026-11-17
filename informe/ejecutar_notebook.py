"""Ejecuta Proyecto3_CUDA.ipynb desde la raiz del repo y guarda las salidas en el mismo archivo."""
import nbformat
from nbclient import NotebookClient
from pathlib import Path
root = Path(__file__).resolve().parent.parent
nb = nbformat.read(root / "Proyecto3_CUDA.ipynb", as_version=4)
NotebookClient(nb, timeout=3000, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, root / "Proyecto3_CUDA.ipynb")
print("ejecutado OK")
