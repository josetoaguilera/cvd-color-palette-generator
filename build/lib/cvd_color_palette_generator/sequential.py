# from aux_functions import *
from skimage.color import delta_e
import numpy as np

from aux_functions import (
    delta_l_matrix,
    get_max_min_indices,
    interpolate_colors_lab_to_lab,
    obtener_posiciones_ordenadas,
    rgb_cmap_to_lab_cmap
)


def sequential_selection(cmap):
    """
    Calculates the indexes of the colors in cmap for a sequential representation.
    Doesn't take into account color vision deficiency.
    Args:
        cmap (list): The color map in rgb.
    Returns:
        result (tuple(int,int)): The indexes of colors in the colormap.
    """

    colores_transformados = []

    def l_fun(a):
        return a / 255
    for i in cmap:
        mapped = list(map(l_fun, i))
        colores_transformados.append(mapped)
    binary_transformed_lab = rgb_cmap_to_lab_cmap(cmap)

    max_index, min_index = get_max_min_indices(binary_transformed_lab)
    return [max_index, min_index]


def sequential_selection_cvd(cmap, cvd):
    """
    Calculates the indexes of the colors in cmap for a sequential representation.
    Based on color vision deficiency (CVD).
    Args:
        cmap (numpy.ndarray): The color map in rgb [0, 255].
        cvd (dict): A dictionary containing color maps for different types of CVD.
    Returns:
        tuple: The selected color pair with the highest diference in luminosity.
    """
    if len(cvd) == 0:
        return sequential_selection(cmap)
    numpy_colors_cvd_list = {}
    for nombre in cvd:
        numpy_colors_cvd_list[nombre] = np.array(cvd[nombre])

    colores_transformados = []
    colores_transformados_cvd_list = {}

    def l_fun(a):
        return a / 255
    for i in cmap:
        mapped = list(map(l_fun, i))
        colores_transformados.append(mapped)

    for nombre in numpy_colors_cvd_list.keys():
        colores_transformados_cvd_list[nombre] = np.divide(
            numpy_colors_cvd_list[nombre], 255)

    cmap_lab = rgb_cmap_to_lab_cmap(cmap)

    cmap_lab_cvd = {}
    for nombre in cvd.keys():
        cmap_lab_cvd[nombre] = rgb_cmap_to_lab_cmap(cvd[nombre])

    delta_l_matrix_orig = delta_l_matrix(cmap_lab)
    pares_cmap_original = obtener_posiciones_ordenadas(delta_l_matrix_orig, 0)

    delta_l_matrix_difference_cvd_list = {}
    pares_cmap_cvd_list = {}
    for nombre in cmap_lab_cvd.keys():
        delta_l_difference_cvd = delta_l_matrix(cmap_lab_cvd[nombre])
        delta_l_matrix_difference_cvd_list[nombre] = delta_l_difference_cvd

        pares_cmap_cvd = obtener_posiciones_ordenadas(
            delta_l_difference_cvd, 0)
        pares_cmap_cvd_list[nombre] = pares_cmap_cvd

    delta_l_index_list = []

    for par in pares_cmap_original:
        cvd_sum = 0
        for par_cmap_cvd in pares_cmap_cvd_list:
            if (par in pares_cmap_cvd_list[par_cmap_cvd] and
                    cvd_sum is not None):
                cvd_sum += delta_l_matrix_difference_cvd_list[
                    par_cmap_cvd][par[0]][par[1]]
            else:
                cvd_sum = None
        if cvd_sum is not None:
            cvd_sum /= len(pares_cmap_cvd_list)
            delta_l_value = delta_l_matrix_orig[
                par[0]][par[1]] * 0.5 + cvd_sum * 0.5
            delta_l_index_list.append([delta_l_value, par])
    delta_l_index_list.sort(reverse=True)
    return delta_l_index_list[0][1]


def sequential_representation_selected(
        cmap, left_luminosity, right_luminosity):
    """
    Calculates the sequential representation of the selected color map.
    Using the given parameters.
    Args:
        cmap (list): The color map in [0,255] rgb.
        left_luminosity (float): the amount of luminosity in the left color.
        right_luminosity (float): the amount of luminosity in the right color.
    Returns:
        numpy.ndarray: The sequential representation of the selected color map in Lab color space.
    """
    cmap_lab = rgb_cmap_to_lab_cmap(cmap)
    left_color_lab = cmap_lab[0]
    right_color_lab = cmap_lab[1]
    left_color_modified = [
       left_luminosity, left_color_lab[1], left_color_lab[2]]
    right_color_modified = [
       right_luminosity, right_color_lab[1], right_color_lab[2]]
    interpolated_colors = interpolate_colors_lab_to_lab(
       [left_color_modified, right_color_modified], 7)
    return np.array(interpolated_colors)


# chequeo secuencial
# diferenciabilidad
# delta_e entre cada variación de color y que sea sumativa
# A B C D
# delta_e(A,B) = X
# delta_e(B,C) = X
# delta_e(C,D) = X
# delta_e(A,C) = 2X
# delta_e(B,D) = 2X
# delta_e(A,D) = 3X
# lumnosidad
# luminosidad distinta de ambos colores
def check_sequential(cmap):
    """
    Check if the given [0, 255] rgb colormap represents sequential values.
    Args:
        cmap (List(int)): The [0, 255] RGB colormap to be checked.
    """
    cmap_lab = rgb_cmap_to_lab_cmap(cmap.copy())

    delta_e_array = {}
    luminosity_check_value = True
    delta_e_check_value = True

    for j in range(1, len(cmap)):
        print("(debería ser aprox: ", j, "veces el valor)", "\n")
        if cmap_lab[j-1][0] <= cmap_lab[j][0]:
            luminosity_check_value = False
        for i in range(len(cmap) - 1):
            # delta_e entre cada variación de color y que sea sumativa
            if i + j < len(cmap):
                print("delta_e posición", i + 1, " y ", i + j + 1, " :")
                delta_e_value = delta_e.deltaE_cie76(
                    cmap_lab[i], cmap_lab[i + j])
                print(round(delta_e_value, 2))
                delta_e_array[j] = delta_e_value
                if delta_e_value < 12:
                    delta_e_check_value = False
        print()
    for j in range(len(cmap)):
        luminosity_value = cmap_lab[j][0]
        print("lumminosidad de color #", j + 1, ": ",
              round(luminosity_value, 2))
    print()
    if delta_e_check_value:
        print("diferenciabilidad correcta")
    else:
        print("diferenciabilidad incorrecta")
    print("luminosidad correcta") if luminosity_check_value else print(
       "luminosidad incorrecta")


def sequential_deviation(cmap, indices, cmap_result):
    """
    Display the sequential deviation between the colors in the colormap.
    Args:
        cmap (list): The [0, 255] RGB color map.
        indices (list): The indices of the colors to consider in the sequential deviation calculation.
        cmap_result (list): The resulting [0, 255] RGB color map.
    """
    sequential_cmap = [cmap[i] for i in indices]
    sequential_cmap_lab = rgb_cmap_to_lab_cmap(sequential_cmap)
    sequential_cmap_result_lab = rgb_cmap_to_lab_cmap(
       [cmap_result[0], cmap_result[-1]])
    delta_e_list = []
    for i in range(len(indices)):
        delta_e_value = delta_e.deltaE_cie76(
           sequential_cmap_lab[i], sequential_cmap_result_lab[i])
        delta_e_list.append(delta_e_value)
    print("desviación izquierda: ", delta_e_list[0])
    print("desviación derecha: ", delta_e_list[1])
