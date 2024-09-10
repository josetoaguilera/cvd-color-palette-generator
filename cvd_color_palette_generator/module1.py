# from preprocess import *
from skimage import io
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def obtener_colores_relevantes(imagen_path, num_colores=20):
    # Cargar la imagen con skimage
    imagen = io.imread(imagen_path)

    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo cargar la imagen en la ruta: {imagen_path}")

    # Obtener las dimensiones de la imagen
    alto, ancho, _ = imagen.shape

    # Aplanar la matriz de píxeles
    pixeles = imagen.reshape((alto * ancho, 3))

    # Aplicar k-means para agrupar los colores
    kmeans = KMeans(n_clusters=num_colores, n_init=10)
    kmeans.fit(pixeles)

    # Obtener los colores dominantes
    colores_dominantes = kmeans.cluster_centers_.astype(int)

    # Contar la frecuencia de cada color
    etiquetas = kmeans.labels_
    frecuencia_colores = Counter(etiquetas)

    # Ordenar los colores por frecuencia
    colores_ordenados = [
        colores_dominantes[i] for i in frecuencia_colores.keys()]

    # Obtener los colores más relevantes
    colores_relevantes = colores_ordenados[:num_colores]

    # Mostrar la imagen seleccionada
    plt.title("Imagen seleccionada")
    plt.xlabel("X píxeles")
    plt.ylabel("Y píxeles")

    image = mpimg.imread(imagen_path)
    plt.imshow(image)
    plt.show()

    return colores_relevantes


def create_choropleth(rgb_colors):
    # Number of colors provided
    num_colors = len(rgb_colors)
    # Convert the RGB list to the required string format
    rgb_colors_str = [f'rgb({r},{g},{b})' for r, g, b in rgb_colors]
    # Sample data
    state_data = {
        'State': ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                  'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA',
                  'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
                  'WV', 'WI', 'WY'],
        'Value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                  110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                  210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
                  310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
                  410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
    }
    # Create a DataFrame
    df = pd.DataFrame(state_data)
    # Create bins and labels based on the number of colors
    bins = np.linspace(df['Value'].min(), df['Value'].max(), num_colors + 1)
    labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(num_colors)]
    # Define discrete color bins and their corresponding colors
    df['Value Category'] = pd.cut(df['Value'],
                                  bins=bins,
                                  labels=labels,
                                  include_lowest=True)
    # Map the provided colors to the bins
    color_map = {label: rgb_colors_str[i] for i, label in enumerate(labels)}
    df['Color'] = df['Value Category'].map(color_map)
    # Custom color scale for the discrete categories
    colorscale = []
    for i in range(num_colors):
        colorscale.append([i / num_colors, rgb_colors_str[i]])
        colorscale.append([(i + 1) / num_colors, rgb_colors_str[i]])
    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['State'],
        z=df['Value'].astype(float),
        locationmode='USA-states',
        colorscale=colorscale,
        colorbar=dict(
            title="Value",
            tickvals=[(bins[i] + bins[i + 1]) / 2 for i in range(num_colors)],
            ticktext=labels
        ),
    ))
    # Update layout for the map
    fig.update_layout(
        title_text='Choropleth Map of USA',
        geo_scope='usa',
    )
    return fig
