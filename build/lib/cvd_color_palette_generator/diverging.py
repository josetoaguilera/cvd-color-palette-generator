# from preprocess import *
# from aux_functions import *
import numpy as np
from pprint import pprint
from colormath.color_objects import LabColor
from skimage.color import delta_e

from aux_functions import (
    complemento_lab,
    delta_l_ratio_matrix,
    filter_tuples_by_exact_numbers,
    get_color_difference_matrix_v2,
    interpolate_colors_lab_to_lab,
    lab_to_rgb,
    obtener_posiciones_ordenadas,
    obtener_posiciones_ordenadas_v2,
    rgb_cmap_to_lab_cmap,
    rgb_to_lab
)


def diverging_selection(cmap):
    """
    Calculates the indexes of the colors in cmap for a diverging representation.
    Doesn't take into account color vision deficiency.
    Args:
        cmap (list): The color map in rgb.
    Returns:
        result (list(int)): The indexes of colors in the colormap.
    """

    cmap_lab = rgb_cmap_to_lab_cmap(cmap)
    binary_matrix_difference = get_color_difference_matrix_v2(cmap_lab)

    pares_cmap_original = obtener_posiciones_ordenadas_v2(binary_matrix_difference, 12)

    lightness_filtered = {}
    for i in range(len(cmap_lab)):
        print(i, cmap_lab[i][0])
        if cmap_lab[i][0] < 50:
            lightness_filtered[i] = cmap_lab[i]

    results_list = filter_tuples_by_exact_numbers(pares_cmap_original, list(lightness_filtered.keys()))
    return results_list[0]


def diverging_selection_cvd(cmap, cvd):
    """
    Calculates the indexes of the colors in cmap for a diverging representation.
    Based on color vision deficiency (CVD).
    Args:
        cmap (numpy.ndarray): The color map in rgb [0, 255].
        cvd (dict): A dictionary containing color maps for different types of CVD.
    Returns:
        tuple: The selected color pair.
    """

    if len(cvd) == 0:
        pares = diverging_selection(cmap)
        return pares
    numpy_colors_cvd_list = {}
    for nombre in cvd:
        numpy_colors_cvd_list[nombre] = np.array(cvd[nombre])

    colores_transformados_cvd_list = {}

    for nombre in numpy_colors_cvd_list.keys():
        colores_transformados_cvd_list[nombre] = np.divide(
           numpy_colors_cvd_list[nombre], 255)

    # rgb a lab
    binary_transformed_lab = rgb_cmap_to_lab_cmap(cmap)

    binary_transformed_lab_cvd = {}
    for nombre in cvd.keys():
        binary_transformed_lab_cvd[nombre] = rgb_cmap_to_lab_cmap(cvd[nombre])

    delta_l_matrix_orig = delta_l_ratio_matrix(binary_transformed_lab)
    pares_cmap_original = obtener_posiciones_ordenadas(delta_l_matrix_orig, 0)

    # matrix delta e v2
    binary_matrix_difference = get_color_difference_matrix_v2(
       binary_transformed_lab)

    pares_cmap_original = obtener_posiciones_ordenadas_v2(
       binary_matrix_difference, 12)

    binary_matrix_difference_cvd_list = {}
    pares_cmap_cvd_list = {}

    for nombre in binary_transformed_lab_cvd.keys():
        binary_matrix_difference_cvd = get_color_difference_matrix_v2(
           binary_transformed_lab_cvd[nombre])
        binary_matrix_difference_cvd_list[nombre] = (
            binary_matrix_difference_cvd
        )

        pares_cmap_cvd = obtener_posiciones_ordenadas_v2(
           binary_matrix_difference_cvd, 12)
        pares_cmap_cvd_list[nombre] = pares_cmap_cvd

    pares_cmap_cvd_list_2 = {}
    cmap_lab_cvd = binary_transformed_lab_cvd.copy()
    delta_l_matrix_difference_cvd_list = {}
    for nombre in cmap_lab_cvd.keys():
        delta_l_difference_cvd = delta_l_ratio_matrix(cmap_lab_cvd[nombre])
        delta_l_matrix_difference_cvd_list[nombre] = delta_l_difference_cvd

        pares_cmap_cvd = obtener_posiciones_ordenadas(
           delta_l_difference_cvd, 0)
        pares_cmap_cvd_list_2[nombre] = pares_cmap_cvd

    delta_e_index_list = []
    for par in pares_cmap_original:
        cvd_sum = 0
        cvd_sum_l = 0
        for par_cmap_cvd in pares_cmap_cvd_list:
            if (par in pares_cmap_cvd_list[par_cmap_cvd] and
                    cvd_sum is not None):
                cvd_sum += binary_matrix_difference_cvd_list[par_cmap_cvd][
                   par[0]][par[1] - 1 - par[0]]
            else:
                cvd_sum = None

        for par_cmap_cvd in pares_cmap_cvd_list_2:
            if (par in pares_cmap_cvd_list_2[par_cmap_cvd] and
                    cvd_sum_l is not None):
                cvd_sum_l += delta_l_matrix_difference_cvd_list[par_cmap_cvd][
                   par[0]][par[1]]
            else:
                cvd_sum_l = None

        if cvd_sum is not None and cvd_sum_l is not None:
            cvd_sum /= len(pares_cmap_cvd_list)
            cvd_sum_l /= len(pares_cmap_cvd_list_2)
            delta_e_value = binary_matrix_difference[par[0]][
               par[1] - 1 - par[0]] * 0.5 + cvd_sum * 0.5
            delta_l_value = delta_l_matrix_orig[par[0]][
               par[1]] * 0.5 + cvd_sum_l * 0.5
            delta_e_index_list.append([delta_l_value, delta_e_value, par])
    delta_e_index_list = sorted(delta_e_index_list, key=lambda x: (x[0], x[1]),
                                reverse=True)
    pprint(delta_e_index_list[:20])
    result_indices = delta_e_index_list[0][2]
    for i in range(len(delta_e_index_list)):
        if delta_e_index_list[i][1] > 0:
            result_indices = delta_e_index_list[i][2]
            break
    return result_indices


def diverging_representation_selected(
      cmap, side_selected, parameter_value, middle_luminosity):
    """
    Calculates the diverging representation of the selected color map.
    Using the given parameters.
    Args:
        cmap (list): The color map in [0,255] rgb.
        side_selected (str): The side of the color map selected ('left' or 'right').
        parameter_value (int): The parameter value for the binary representation.
        middle_luminosity (int): The luminosity value for the middle color.
    Returns:
        numpy.ndarray: The diverging representation of the selected color map in Lab color space.
    """
    selected_position = 0
    if side_selected == "left":
        selected_position = 0
    elif side_selected == "right":
        selected_position = 1
    cmap_lab = rgb_cmap_to_lab_cmap(cmap)
    selected_color_lab = cmap_lab[selected_position]
    oposite_color_lab = [selected_color_lab[0],
                         selected_color_lab[1] * -1,
                         selected_color_lab[2] * -1]
    other_color_lab = cmap_lab[1-selected_position]
    oposite_parameter = interpolate_colors_lab_to_lab(
        [other_color_lab, oposite_color_lab], 100)[parameter_value-1]

    binary_cmap_lab = [None, None]
    binary_cmap_lab[selected_position] = cmap_lab[selected_position]
    binary_cmap_lab[1-selected_position] = oposite_parameter

    binary_cmap_lab_1_l = binary_cmap_lab[0][0]
    binary_cmap_lab_2_l = binary_cmap_lab[1][0]

    lightness_correction = False

    if binary_cmap_lab_1_l > 50 and binary_cmap_lab_2_l > 50:
        lightness_correction = True
        binary_cmap_lab_1_l = 50.
        binary_cmap_lab_2_l = 50.
    elif binary_cmap_lab_1_l > 50 and binary_cmap_lab_2_l <= 50:
        binary_cmap_lab_1_l = binary_cmap_lab_2_l
        lightness_correction = True
    elif binary_cmap_lab_2_l > 50 and binary_cmap_lab_1_l <= 50:
        binary_cmap_lab_2_l = binary_cmap_lab_1_l
        lightness_correction = True

    new_binary_cmap_lab = [binary_cmap_lab_1_l,
                           binary_cmap_lab[0][1],
                           binary_cmap_lab[0][2]], [
                               binary_cmap_lab_2_l,
                               binary_cmap_lab[1][1],
                               binary_cmap_lab[1][2]]

    interpolated_colors = interpolate_colors_lab_to_lab(
        np.array(new_binary_cmap_lab), n_colors=3)

    middle_color = interpolated_colors[1]

    middle_color[0] = middle_luminosity
    interpolated_colors[1] = middle_color

    n_interpolated_colors = 7
    result_lab = interpolate_colors_lab_to_lab(interpolated_colors,
                                               n_interpolated_colors)
    if not lightness_correction:
        return result_lab
    else:
        result_rgb = [
            lab_to_rgb(LabColor(c[0], c[1], c[2])) for c in result_lab]
        result_rgb = np.multiply(result_rgb, 255)
        new_result_lab = [rgb_to_lab(c) for c in result_rgb]
        result_cmap = [
            [lab.lab_l, lab.lab_a, lab.lab_b] for lab in new_result_lab]
        return result_cmap


# chequeo radial
# diferenciabilidad
# similar a sequential pero con un punto neutro en el centro
# luminosidad menor en el centro y
# otras luminosidades mayores en los extremos y de magnitudes similares
def check_diverging(cmap):
    """
    Check if the given [0, 255] rgb colormap represents diverging values.
    Args:
        cmap (List(int)): The [0, 255] RGB colormap to be checked.
    """
    luminosity_check_value = True
    cmap_size = len(cmap)
    cmap_lab = rgb_cmap_to_lab_cmap(cmap.copy())
    binary_cmap = [cmap_lab[0], cmap_lab[-1]]

    # delta_e ambos colores
    delta_e_value_original = delta_e.deltaE_cie76(cmap_lab[0], cmap_lab[-1])
    print("delta_e: color izquierda con color de la derecha:")
    print(round(delta_e_value_original, 2))
    # delta_e opuesto izq y derecho
    print("delta_e: color izquierda con su opuesto:")
    print(round(delta_e.deltaE_cie76(
        cmap_lab[0], complemento_lab(cmap_lab[0])), 2))
    # delta_e opuesto der y izq
    print("delta_e: color derecha con su opuesto:")
    print(round(delta_e.deltaE_cie76(
        cmap_lab[-1], complemento_lab(cmap_lab[-1])), 2))
    print()
    if delta_e_value_original > 12:
        print("diferenciabilidad de colores laterales correcta")
    else:
        print("diferenciabilidad de colores laterales incorrecta")
    print()

    left_luminosity = binary_cmap[0][0]
    right_luminosity = binary_cmap[-1][0]
    middle_index = int(cmap_size / 2)
    middle_luminosity = cmap_lab[middle_index][0]

    for j in range(len(cmap)):
        luminosity_value = cmap_lab[j][0]
        print("lumminosidad de color #", j + 1, ": ",
              round(luminosity_value, 2))
        if (j < middle_index and luminosity_value > middle_luminosity and
                luminosity_value < left_luminosity):
            luminosity_check_value = False
        if (j > middle_index and luminosity_value > middle_luminosity and
                luminosity_value < right_luminosity):
            luminosity_check_value = False

    print()

    if luminosity_check_value:
        print("luminosidad de colores correcta")
    else:
        print("luminosidad de colores incorrecta")


def diverging_deviation(cmap, indices, cmap_result):
    """
    Display the diverging deviation in the colormap.
    Args:
        cmap (list): The [0, 255] RGB color map.
        indices (list): The indices of the colors to consider in the diverging deviation calculation.
        cmap_result (list): The resulting [0, 255] RGB color map.
    """
    radial_cmap = [cmap[i] for i in indices]
    radial_cmap_lab = rgb_cmap_to_lab_cmap(radial_cmap)
    radial_cmap_result_lab = rgb_cmap_to_lab_cmap(
        [cmap_result[0], cmap_result[-1]])
    delta_e_list = []
    for i in range(len(indices)):
        delta_e_value = delta_e.deltaE_cie76(
            radial_cmap_lab[i], radial_cmap_result_lab[i])
        delta_e_list.append(delta_e_value)
    print("desviación izquierda: ", delta_e_list[0])
    print("desviación derecha: ", delta_e_list[1])
