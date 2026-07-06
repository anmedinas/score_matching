"""Descarga MNIST via torchvision y lo deja disponible para train.py y train_clf.py.

La ruta de destino se fija aqui y debe coincidir con la que usan los scripts
de entrenamiento. Los datos descargados no se incluyen en la entrega, solo
este script.
"""

from torchvision.datasets import MNIST

DATA_DIR = "data"


def download_data(root: str = DATA_DIR) -> None:
    MNIST(root=root, train=True, download=True)
    MNIST(root=root, train=False, download=True)


if __name__ == "__main__":
    download_data()
