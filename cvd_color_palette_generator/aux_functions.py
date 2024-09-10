# from preprocess import *
import numpy as np
from skimage import color
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from skimage.color import delta_e

# AUXILIAR


def get_max_min_indices(lst):
    """
    Calculates the index of the max and min values in the first position of a list of lists.

    Args:
        lst ([[int]]): list of lists.

    Returns:
        A tuple containing, respectively, an int (max value index) and a int (min value index).
    """
    if not lst:
        return None, None

    max_index = min_index = 0
    max_value = min_value = lst[0][0]

    for i in range(1, len(lst)):
        if lst[i][0] > max_value:
            max_value = lst[i][0]
            max_index = i
        if lst[i][0] < min_value:
            min_value = lst[i][0]
            min_index = i

    return max_index, min_index

# media_puntos

# show_simulated_image


# ESPACIO DE COLORES


# rgb_cmap_to_lab_cmap
def rgb_cmap_to_lab_cmap(cmap):
    """
    Converts a [0,255] RGB colormap to a numpy LAB colormap.
    Args:
        cmap (List[Tuple[int, int, int]]): The RGB colormap to be converted in [0,255] format.
    Returns:
        List[Tuple[float, float, float]]: The LAB colormap.
    """

    colores_lab = []
    for c in cmap:
        rgb_color = np.array(c, dtype=np.uint8)
        lab_color = color.rgb2lab(rgb_color)
        colores_lab.append(lab_color)
    return colores_lab


# lab_cmap_to_rgb_cmap
def lab_cmap_to_rgb_cmap(cmap_lab):
    """
    Converts a colormap from LAB color space to RGB color space.
    Args:
        cmap_lab (list): List of colors in LAB color space.
    Returns:
        list: List of colors in RGB color space.
    """

    colores_rgb = []
    for c in cmap_lab:
        rgb_color = color.lab2rgb(c)
        colores_rgb.append(rgb_color)
    return colores_rgb


# rgb_to_lab
def rgb_to_lab(rgb):
    """
    Converts an RGB color to the colormath Lab color space.
    Args:
        rgb (tuple): A tuple representing the RGB color values.
    Returns:
        LabColor: The color converted to the Lab color space.
    """

    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return convert_color(sRGBColor(r, g, b, is_upscaled=True), LabColor)


# lab_to_rgb
def lab_to_rgb(lab):
    """
    Converts a color from the colormath CIELAB color space to the [0,1] numpy RGB color space.
    Args:
        lab (tuple): A tuple representing the CIELAB color values.
    Returns:
        list: A list containing the RGB color values in the sRGB color space.
    """

    rgb = convert_color(lab, sRGBColor)
    return [
        np.clip(rgb.rgb_r, 0, 1),
        np.clip(rgb.rgb_g, 0, 1),
        np.clip(rgb.rgb_b, 0, 1),
    ]


# show_color


# show_colors
def show_colors(color_list, axis_state="on"):
    """
    Displays a grid of colors.
    Args:
        color_list (list): A list of [0, 255] RGB color values.
        axis_state (str, optional): The state of the axis. Defaults to "on".
    Returns:
        None
    """

    color_quantity = len(color_list)
    _, ax = plt.subplots()
    flag = np.empty((1, color_quantity, 3), dtype=np.uint8)
    for color_index in range(len(color_list)):
        flag[0, color_index, :] = color_list[color_index]
    ax.axis(axis_state)
    plt.imshow(flag)
    plt.show()


# interpolate_colors_rgb_to_lab

# interpolate_colors_lab_to_rgb


# interpolate_colors_lab_to_lab
def interpolate_colors_lab_to_lab(colors, n_colors=256):
    """
    Interpolates colors in Lab color space.
    Args:
        colors (List[Tuple[float, float, float]]): List of Lab colors to interpolate.
        n_colors (int, optional): Number of colors to interpolate. Defaults to 256.
    Returns:
        np.ndarray: Interpolated colors in Lab color space.
    """

    colors_lab = colors.copy()

    # Crear arrays de cada componente L, a, b
    l_values = np.array([c[0] for c in colors_lab])
    a_values = np.array([c[1] for c in colors_lab])
    b_values = np.array([c[2] for c in colors_lab])

    # Interpolación lineal de cada componente
    x_orig = np.linspace(0, 1, len(colors))
    x_new = np.linspace(0, 1, n_colors)

    l_interp = np.interp(x_new, x_orig, l_values)
    a_interp = np.interp(x_new, x_orig, a_values)
    b_interp = np.interp(x_new, x_orig, b_values)

    # Convertir colores interpolados de Lab a RGB
    interpolated_colors = np.zeros((n_colors, 3))
    for i, (l, a, b) in enumerate(zip(l_interp, a_interp, b_interp)):
        lab_color = LabColor(l, a, b)
        lab_color = np.array([l, a, b])
        interpolated_colors[i] = lab_color

    return interpolated_colors


# interpolate_colors_lab


# complemento_lab
# devuelve el complemento en cielab numpy del color cielab numpy
def complemento_lab(lab_color):
    """
    Calculates the complement of a CIELAB color.
    Args:
        lab_color (list(float, float, float)): A list containing the L, a, and b values of the CIELAB color.
    Returns:
        list(float, float, float): A list containing the complement of the input CIELAB color.
    """

    l_value = lab_color[0]
    a_value = lab_color[1]
    b_value = lab_color[2]
    return [100 - l_value, a_value * -1, b_value * -1]


# MATRIX


# get_indexes_below_n
def get_indexes_below_n(matrix, n):
    """
    Returns a list of indexes of elements in the matrix that are below the given value.
    Args:
        matrix (list): The matrix to search for elements.
        n (int): The value to compare elements against.
    Returns:
        list(tuple(int, int)): A list of the indexes of elements below the given value.
    """

    indexes = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value < n:
                indexes.append((i, j))
    return indexes


# get_color_difference_matrix
def get_color_difference_matrix(cmap):
    """
    Calculates the color difference matrix for a given colormap.
    Args:
        cmap (list): A list of colors representing the colormap.
    Returns:
        list: A matrix containing the color difference values between each pair of colors in the colormap.
    """

    matrix_delta_e = []
    for c in cmap:
        row = []
        for c2 in cmap:
            result_delta_e = delta_e.deltaE_cie76(c, c2)
            row.append(result_delta_e)
        matrix_delta_e.append(row)

    return matrix_delta_e


# get_color_difference_matrix_v2
def get_color_difference_matrix_v2(cmap):
    """
    Calculates the color difference matrix for a given colormap.
    Excluding inner diagonal and upper triangle.
    Args:
        cmap (list): A list of colors representing the colormap.
    Returns:
        list: A matrix containing the color difference values between each pair of colors in the colormap.
    """

    matrix_delta_e = []
    cmap_size = len(cmap)
    for i in range(cmap_size):
        row = []
        for j in range(cmap_size - i - 1):
            result_delta_e = delta_e.deltaE_cie76(
                cmap[i], cmap[cmap_size - j - 1]
            )
            row.append(result_delta_e)
        row.reverse()
        matrix_delta_e.append(row)

    return matrix_delta_e


# get_similar_colors_index
def get_similar_colors_index(matrix, min_value=13):
    """
    Returns the indexes of similar colors in a matrix.
    Args:
        matrix (list): The matrix containing color values.
        min_value (int, optional): The minimum color value to consider as similar. Defaults to 13.
    Returns:
        list: A list of indexes representing similar colors in the matrix.
    """

    result = get_indexes_below_n(matrix, min_value)
    return [[x, y] for x, y in result if x != y]


# find_max_index_in_list_of_lists
def find_max_index_in_list_of_lists(list_of_lists):
    """
    Finds the maximum value in a list of lists and returns its index.
    Args:
        list_of_lists (list of lists): A list containing sublists with numerical values.
    Returns:
        max_index (tuple(int, int)): A tuple (outer_index, inner_index) representing the index of the maximum value.
    """
    if not list_of_lists:
        return None

    max_value = float("-inf")
    max_index = (-1, -1)

    for outer_index, sublist in enumerate(list_of_lists):
        for inner_index, value in enumerate(sublist):
            if value > max_value:
                max_value = value
                max_index = (outer_index, inner_index)

    return max_index


# index_max_delta_e

# get_values_at_indexes


# split_universe
# retorna lista con los grupos de indices de los colores que no son similares
def split_universe(u_list, split_index):
    """
    Splits the given universe list based on the provided split index.
    The splitted universe list contains sublists where each sublist can contain only one element from each split index.
    Args:
        u_list (list): The universe list to be split.
        split_index (list(list)): The list of indexes used to split the universe list.
    Returns:
        list: A list of sublists, where each sublist represents a split of the universe list.
    Example:
        >>> u_list = [1, 2, 3, 4, 5]
        >>> split_index = [[1, 3], [2, 4]]
        >>> split_universe(u_list, split_index)
        [[3, 4, 5], [2, 3, 5], [1, 4, 5], [1, 2, 5]]
    """
    split_universe = []
    result = []
    for indices in split_index:
        if len(split_universe) == 0:
            result.append([indices[0]])
            result.append([indices[1]])
            split_universe = result.copy()
        else:
            for s in split_universe:
                if indices[0] in s or indices[1] in s:
                    continue
                else:
                    result.remove(s)
                    s1 = s.copy()
                    s1.append(indices[0])
                    result.append(s1)
                    s2 = s.copy()
                    s2.append(indices[1])
                    result.append(s2)
            split_universe = result.copy()
    result = []
    for i in split_universe:
        row = u_list.copy()
        for j in u_list:
            if j in i:
                row.remove(j)
        result.append(row)
    return result


# obtener_posiciones_ordenadas
def obtener_posiciones_ordenadas(matrix, n):
    """
    Obtains the positions of values greater than n in a matrix and returns them in descending order of their values.
    Args:
        matrix (list[list[int]]): The matrix to search for positions.
        n (int): The threshold value.
    Returns:
        list[tuple[int, int]]: A list of tuples representing the positions (row, column) of values greater than n, sorted in descending order of their values.
    """

    posiciones = []

    # Recorre la matriz y almacena las posiciones de los valores mayores a n
    for i, fila in enumerate(matrix):
        for j, valor in enumerate(fila):
            if valor > n:
                posiciones.append((i, j, valor))

    # Ordena las posiciones por el valor de mayor a menor
    posiciones.sort(key=lambda x: x[2], reverse=True)

    # Devuelve solo las posiciones (fila, columna)
    return [(i, j) for i, j, valor in posiciones]


# obtener_posiciones_ordenadas_v2
def obtener_posiciones_ordenadas_v2(matrix, n):
    """
    Returns a list of positions (row, column) of values in the matrix that are greater than n, 
    sorted in descending order based on their values.
    Suited for the optimized version of the matrix.
    Args:
        matrix (list[list[int]]): The matrix to search for values.
        n (int): The threshold value.
    Returns:
        list[tuple[int, int]]: A list of positions (row, column) of values greater than n, 
        sorted in descending order based on their values.
    """

    posiciones = []

    # Recorre la matriz y almacena las posiciones de los valores mayores a n
    for i, fila in enumerate(matrix):
        for j, valor in enumerate(fila):
            if valor > n:
                posiciones.append((i, j, valor))

    # Ordena las posiciones por el valor de mayor a menor
    posiciones.sort(key=lambda x: x[2], reverse=True)

    # Devuelve solo las posiciones (fila, columna)
    return [(i, i + j + 1) for i, j, valor in posiciones]


# delta_l_matrix
def delta_l_matrix(cmap_lab):
    """
    Calculates the delta L matrix based on the given cmap_lab.
    delta L is the difference in luminance between two colors.
    Args:
        cmap_lab (list): A list of LAB color values.
    Returns:
        list: The delta L matrix.
    """

    matrix = []
    for i in range(len(cmap_lab)):
        row = []
        for j in range(len(cmap_lab)):
            delta_l = cmap_lab[i][0] - cmap_lab[j][0]
            row.append(delta_l)
        matrix.append(row)
    return matrix


# filter_tuples_by_exact_numbers
def filter_tuples_by_exact_numbers(tuple_array, number_array):
    """
    Filters a list of tuples based on the presence of exact numbers.
    Args:
        tuple_array (list): A list of tuples.
        number_array (list): A list of numbers.
    Returns:
        list: A filtered list of tuples where each tuple contains only numbers present in the number_array.
    Example:
        >>> tuple_array = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        >>> number_array = [2, 3, 5, 4, 6, 7]
        >>> filter_tuples_by_exact_numbers(tuple_array, number_array)
        [(4, 5, 6)]
    """
    filtered_tuples = [t for t in tuple_array if all(num in number_array for num in t)]
    return filtered_tuples


# delta_l_ratio_matrix
def delta_l_ratio_matrix(cmap_lab):
    """
    Calculates the delta L ratio matrix based on the given cmap_lab.
    delta L ratio matrix is a matrix that tells how many times the luminance of two colors is less than 50.
    Args:
        cmap_lab (list): A list of Lab colors.
    Returns:
        list: The delta L ratio matrix.
    """
    matrix = []
    for i in range(len(cmap_lab)):
        row = []
        for j in range(len(cmap_lab)):
            delta_l = (
                2
                if (cmap_lab[i][0] < 50 and cmap_lab[j][0] < 50)
                else 1 if (cmap_lab[j][0] < 50 or cmap_lab[i][0] < 50) else 0
            )
            row.append(delta_l)
        matrix.append(row)
    return matrix
