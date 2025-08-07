# CVD Color Palette Generator

Una librería de Python para generar paletas de colores accesibles para personas con discapacidad visual por deficiencia de color (CVD - Color Vision Deficiency).

## Descripción

Esta librería permite crear paletas de colores optimizadas que sean distinguibles tanto para personas con visión normal como para aquellas con diferentes tipos de daltonismo. La herramienta incluye simuladores de diferentes tipos de CVD y algoritmos para optimizar la diferenciabilidad de colores en múltiples representaciones visuales.

## Características Principales

- **Múltiples tipos de representación de colores:**
  - **Binario**: Paletas de dos colores con máximo contraste
  - **Secuencial**: Gradientes ordenados para datos continuos
  - **Categórico**: Conjuntos de colores distinguibles para categorías discretas
  - **Divergente**: Paletas con punto central neutro para datos que divergen

- **Simulación de CVD**: Compatible con múltiples simuladores (Brettel1997, Vienot1999, Machado2009, Vischeck, CoblisV1, CoblisV2)

- **Tipos de daltonismo soportados:**
  - Protanopia (deficiencia de rojo)
  - Deuteranopia (deficiencia de verde)
  - Tritanopia (deficiencia de azul)

- **Análisis de calidad**: Verificación automática de diferenciabilidad usando métricas Delta-E y luminosidad

## Instalación

### Desde el código fuente

```bash
git clone https://github.com/josetoaguilera/cvd-color-palette-generator
cd cvd-color-palette-generator
pip install -e .
```

### Dependencias

Las dependencias se instalarán automáticamente, pero incluyen:

- colormath>=3.0.0
- daltonlens>=0.1.5
- matplotlib>=3.9.0
- scikit-image>=0.24.0
- plotly>=5.23.0
- pandas>=2.2.2
- ipywidgets==7.7.1
- opencv-python>=4.10.0.84
- scikit-learn>=1.5.1
- seaborn>=0.13.2

## Uso Básico

### Importar la librería

```python
from cvd_color_palette_generator.binary import binary_selection, binary_selection_cvd
from cvd_color_palette_generator.sequential import sequential_selection, sequential_selection_cvd
from cvd_color_palette_generator.categorical import categorical_selection, categorical_selection_cvd
from cvd_color_palette_generator.diverging import diverging_selection, diverging_selection_cvd
from cvd_color_palette_generator.aux_functions import show_colors, lab_cmap_to_rgb_cmap
```

### Ejemplo: Paleta Binaria

```python
import numpy as np
from daltonlens import simulate

# Cargar una imagen o definir una paleta de colores
cmap = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB

# Simular CVD
simulator = simulate.Simulator_Brettel1997()
im_cmap = np.asarray([cmap])
cvd = {}
cvd["protan"] = simulator.simulate_cvd(im_cmap, deficiency=simulate.Deficiency.PROTAN, severity=1)[0]
cvd["deutan"] = simulator.simulate_cvd(im_cmap, deficiency=simulate.Deficiency.DEUTAN, severity=1)[0]

# Seleccionar colores binarios optimizados para CVD
indices = binary_selection_cvd(cmap, cvd)
print(f"Índices seleccionados: {indices}")
```

### Ejemplo: Paleta Secuencial

```python
# Seleccionar colores para representación secuencial
indices = sequential_selection_cvd(cmap, cvd)
luminosity_left = 20.0
luminosity_right = 80.0

# Generar representación secuencial
result_cmap_lab = sequential_representation_selected([cmap[i] for i in indices], 
                                                   luminosity_left, 
                                                   luminosity_right)
result_cmap = lab_cmap_to_rgb_cmap(result_cmap_lab)
```

### Ejemplo: Paleta Categórica

```python
# Para datos categóricos con múltiples clases
min_delta_e = 12  # Diferencia mínima perceptual entre colores
indices = categorical_selection_cvd(cmap, cvd, min_delta_e)
```

## Uso Avanzado con Jupyter Notebook

La librería incluye widgets interactivos para Jupyter Notebook que permiten experimentar con diferentes parámetros:

```python
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt

# Función interactiva para ajustar parámetros de paleta binaria
def interactive_binary_palette(cmap, cvd, simulator):
    interact(set_binary_parameters,
             cmap=widgets.fixed(cmap),
             cvd=widgets.fixed(cvd),
             fixed_side=widgets.ToggleButtons(
                 options=['left', 'right'], 
                 description='Lado fijo:'
             ),
             parameter=widgets.IntSlider(
                 value=1, min=1, max=100, step=1, 
                 description='Parámetro:'
             ),
             simulator=widgets.fixed(simulator))
```

## Análisis de Calidad

La librería incluye funciones para verificar la calidad de las paletas generadas:

```python
from cvd_color_palette_generator.binary import check_binary
from cvd_color_palette_generator.sequential import check_sequential
from cvd_color_palette_generator.categorical import check_categorical
from cvd_color_palette_generator.diverging import check_diverging

# Verificar paleta binaria
check_binary(result_cmap)

# Verificar paleta secuencial  
check_sequential(result_cmap)

# Verificar paleta categórica
check_categorical(result_cmap)

# Verificar paleta divergente
check_diverging(result_cmap)
```

## Visualización y Exportación

### Mostrar colores

```python
from cvd_color_palette_generator.aux_functions import show_colors

# Visualizar la paleta generada
show_colors(result_cmap, axis_state='off')
```

### Crear mapas coropléticos

```python
from cvd_color_palette_generator.module1 import create_choropleth

# Crear visualización cartográfica
fig = create_choropleth(result_cmap)
fig.show()
```

### Usar con Matplotlib y Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Crear colormap personalizado para matplotlib
custom_cmap = ListedColormap(result_cmap)

# Usar en gráficos
plt.scatter(x, y, c=values, cmap=custom_cmap)
plt.colorbar()
plt.show()

# Usar con seaborn
sns.heatmap(data, cmap=custom_cmap)
plt.show()
```

## Estructura del Proyecto

```
cvd_color_palette_generator/
├── __init__.py
├── aux_functions.py      # Funciones auxiliares y utilidades
├── binary.py            # Paletas binarias
├── categorical.py       # Paletas categóricas  
├── diverging.py         # Paletas divergentes
├── sequential.py        # Paletas secuenciales
├── module1.py          # Funciones de visualización
└── module2.py          # Funciones adicionales
```

## Casos de Uso

### Visualización Científica
- Mapas de calor en investigación
- Gráficos de datos experimentales
- Representaciones de imágenes médicas

### Cartografía y GIS
- Mapas coropléticos
- Visualización de datos geoespaciales
- Sistemas de información geográfica

### Análisis de Datos
- Dashboards accesibles
- Reportes empresariales
- Visualizaciones estadísticas

### Diseño Web y UI
- Interfaces accesibles
- Gráficos web interactivos
- Aplicaciones móviles

## Fundamentos Técnicos

### Espacios de Color
- Conversión RGB ↔ LAB para cálculos perceptuales
- Métricas Delta-E (CIE76) para diferencias de color
- Optimización de luminosidad para accesibilidad

### Algoritmos de Selección
- Matriz de diferencias de color para optimización
- Filtrado por similitud perceptual
- Balanceado de características para CVD y visión normal

### Simulación CVD
- Múltiples modelos de simulación disponibles
- Transformaciones matriciales para diferentes tipos de daltonismo
- Validación cruzada con diferentes severidades

## Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Autor

**Jose Tomas Aguilera Yevenes**
- Email: joseaguilera@ug.uchile.cl
- GitHub: [@josetoaguilera](https://github.com/josetoaguilera)

## Citas y Referencias

Si usas esta librería en tu investigación, por favor considera citar:

```bibtex
@software{cvd_color_palette_generator,
  author = {Aguilera Yevenes, Jose Tomas},
  title = {CVD Color Palette Generator: A Python Library for Accessible Color Palettes},
  url = {https://github.com/josetoaguilera/cvd-color-palette-generator},
  year = {2024}
}
```

## Agradecimientos

- Comunidad de DaltonLens por las herramientas de simulación CVD
- Investigadores en accesibilidad visual y percepción de color
- Comunidad de código abierto de Python para visualización científica

---

**Nota**: Esta librería está en desarrollo activo. Las APIs pueden cambiar en versiones futuras. Se recomienda fijar la versión en proyectos de producción.