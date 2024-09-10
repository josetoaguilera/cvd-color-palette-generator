# from aux_functions import *
import json
import matplotlib.colors as mcolors

from module1 import obtener_colores_relevantes


def save_and_load_image(uploaded_file, filename, num_colores=20):
    with open("./" + filename, "wb") as fp:
        fp.write(uploaded_file)

    try:
        colores_relevantes = obtener_colores_relevantes(filename, num_colores=num_colores)
        return colores_relevantes
    except Exception as e:
        print(f"Error: {e}")


# Save the colormap data
def save_colormap(colores):
    new_colores = [(r, g, b) for r, g, b in colores]
    colormap_data = {'colores': new_colores}
    with open('custom_cmap.json', 'w') as f:
        json.dump(colormap_data, f)


# Load the colormap data and recreate the colormap
def load_colormap():
    try:
        with open('custom_cmap.json', 'r') as f:
            colormap_data = json.load(f)
            return mcolors.ListedColormap(colormap_data['colores'],
                                          name='custom_cmap')
    except Exception as e:
        print(f"Error: {e}")
        return 'Accent'


# Save the colormap data
def save_colorpalette(colores):
    new_colores = [(int(r), int(g), int(b)) for r, g, b in colores]
    colormap_data = {'original': new_colores}
    with open('color_palette.json', 'w') as f:
        json.dump(colormap_data, f)


def save_cvd_colorpalette(colores, cvd_name):
    new_colores = [(int(r), int(g), int(b)) for r, g, b in colores]
    with open('color_palette.json', 'r') as f:
        cvd_data = json.load(f)
    cvd_data[cvd_name] = new_colores
    with open('color_palette.json', 'w') as f:
        json.dump(cvd_data, f)


# Load the colormap data and recreate the colormap
def load_colorpalette():
    try:
        with open('color_palette.json', 'r') as f:
            colormap_data = json.load(f)
            return colormap_data
    except Exception as e:
        print(f"Error: {e}")
        return {}