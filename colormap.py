import os
import numpy as np
import matplotlib.pyplot as plt
import glob

Nx = 32
Ny = 16
Lx = 2.
Ly = 1.

# Crear la carpeta "images" si no existe
if not os.path.exists("images"):
    os.makedirs("images")

# Cargar y visualizar los archivos guardados
file_list = sorted(glob.glob("data/temperature_*.txt"))

for file in file_list:
    data = np.loadtxt(file)
    plt.imshow(data, origin='lower', cmap='hot', interpolation='nearest',extent=[0, Lx, 0, Ly])
    plt.colorbar(label='Temperature')
    #plt.title(f'Temperature Map from {file}')
    filename = os.path.join("images", os.path.basename(file))  # Carpeta "images" + nombre del archivo
    plt.savefig(f'{filename}.png')  # Guardar como imagen en la carpeta "images"
    plt.clf()
