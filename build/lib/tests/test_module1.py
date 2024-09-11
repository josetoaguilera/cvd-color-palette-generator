from cvd_color_palette_generator.module1 import *

# URL de la imagen
imagen_url = 'https://live.staticflickr.com/4205/35866222205_9fdf4e5598_b.jpg'

# Abrir la URL y leer la imagen en un objeto BytesIO
with urllib.request.urlopen(imagen_url) as url:
    image_data = url.read()
    image = Image.open(BytesIO(image_data))

    # Guardar la imagen en un archivo temporal
    temp_image_path = 'temp_image.jpg'
    image.save(temp_image_path)

    # Usar la función con la ruta del archivo temporal
    try:
        colores_relevantes = obtener_colores_relevantes(temp_image_path)
        pprint(colores_relevantes)
        show_colors(colores_relevantes, axis_state='off')
    except Exception as e:
        print(f"Error: {e}")

cmap = colores_relevantes.copy()