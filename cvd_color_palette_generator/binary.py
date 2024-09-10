from aux_functions import (
    rgb_cmap_to_lab_cmap,
    get_color_difference_matrix,
    find_max_index_in_list_of_lists,
    interpolate_colors_lab_to_lab,
    complemento_lab,
    get_color_difference_matrix_v2,
    obtener_posiciones_ordenadas_v2,
)
import numpy as np
from skimage.color import delta_e


def binary_selection(cmap):
    """
    Calculates the indexes of the colors in cmap for a binary representation.
    Doesn't take into account color vision deficiency.
    Args:
        cmap (list): The color map in rgb.
    Returns:
        binary_max_index (tuple(int,int)): The indexes of colors in the colormap.
    """

    colores_transformados = []

    def l_fun(a):
        return a / 255
    for i in cmap:
        mapped = list(map(l_fun, i))
        colores_transformados.append(mapped)

    binary_transformed_lab = rgb_cmap_to_lab_cmap(cmap)
    binary_matrix_difference = get_color_difference_matrix(
        binary_transformed_lab)
    binary_max_index = find_max_index_in_list_of_lists(
        binary_matrix_difference)

    return binary_max_index


def binary_selection_cvd(cmap, cvd):
    """
    Calculates the indexes of the colors in cmap for a binary representation.
    Based on color vision deficiency (CVD).
    Args:
        cmap (numpy.ndarray): The color map in rgb [0, 255].
        cvd (dict): A dictionary containing color maps for different types of CVD.
    Returns:
        tuple: The selected color pair with the highest delta E value.
    """


    if len(cvd) == 0:
        return binary_selection(cmap)
    numpy_colors_cvd_list = {}
    for nombre in cvd:
        numpy_colors_cvd_list[nombre] = np.array(cvd[nombre])

    colores_transformados_cvd_list = {}

    for nombre in numpy_colors_cvd_list.keys():
        colores_transformados_cvd_list[nombre] = np.divide(
            numpy_colors_cvd_list[nombre], 255
        )

    # rgb a lab
    binary_transformed_lab = rgb_cmap_to_lab_cmap(cmap)

    binary_transformed_lab_cvd = {}
    for nombre in cvd.keys():
        binary_transformed_lab_cvd[nombre] = rgb_cmap_to_lab_cmap(cvd[nombre])

    # matrix delta e v2
    binary_matrix_difference = get_color_difference_matrix_v2(
        binary_transformed_lab)

    pares_cmap_original = obtener_posiciones_ordenadas_v2(
        binary_matrix_difference, 12)

    binary_matrix_difference_cvd_list = {}
    pares_cmap_cvd_list = {}
    for nombre in binary_transformed_lab_cvd.keys():
        binary_matrix_difference_cvd = get_color_difference_matrix_v2(
            binary_transformed_lab_cvd[nombre]
        )
        binary_matrix_difference_cvd_list[nombre] = (
            binary_matrix_difference_cvd)

        pares_cmap_cvd = obtener_posiciones_ordenadas_v2(
            binary_matrix_difference_cvd, 12
        )
        pares_cmap_cvd_list[nombre] = pares_cmap_cvd

    delta_e_index_list = []
    for par in pares_cmap_original:
        cvd_sum = 0
        for par_cmap_cvd in pares_cmap_cvd_list:
            if (par in pares_cmap_cvd_list[par_cmap_cvd]
                    and cvd_sum is not None):
                cvd_sum += binary_matrix_difference_cvd_list[
                    par_cmap_cvd][par[0]][par[1] - 1 - par[0]]
            else:
                cvd_sum = None
        if cvd_sum is not None:
            cvd_sum /= len(pares_cmap_cvd_list)
            delta_e_value = (
                binary_matrix_difference[par[0]][par[1] - 1 - par[0]] * 0.5
                + cvd_sum * 0.5
            )
            delta_e_index_list.append([delta_e_value, par])
    delta_e_index_list.sort(reverse=True)
    return delta_e_index_list[0][1]


def binary_representation_selected(cmap, side_selected, parameter_value):
    """
    Calculates the binary representation of the selected color map.
    Using the given parameters.
    Args:
        cmap (list): The color map in [0,255] rgb.
        side_selected (str): The side of the color map selected ('left' or 'right').
        parameter_value (int): The parameter value for the binary representation.
    Returns:
        numpy.ndarray: The binary representation of the selected color map in Lab color space.
    """
    selected_position = 0
    if side_selected == "left":
        selected_position = 0
    elif side_selected == "right":
        selected_position = 1
    cmap_lab = rgb_cmap_to_lab_cmap(cmap)
    selected_color_lab = cmap_lab[selected_position]
    oposite_color_lab = [
        100 - selected_color_lab[0],
        selected_color_lab[1] * -1,
        selected_color_lab[2] * -1,
    ]
    other_color_lab = cmap_lab[1 - selected_position]
    oposite_parameter = interpolate_colors_lab_to_lab(
        [other_color_lab, oposite_color_lab],
        100
        )[parameter_value - 1]

    result_lab = [None, None]
    result_lab[selected_position] = cmap_lab[selected_position]
    result_lab[1 - selected_position] = oposite_parameter

    numpy_colors = np.array(result_lab)
    return numpy_colors


# chequeo binario
# diferenciabilidad
# delta_e entre ambos colores
# delta_e entre el opuesto del color izq y el color derecho
# delta_e entre el opuesto del color der y el color izq
def check_binary(cmap):
    """
    Check if the given [0, 255] rgb colormap represents binary values.
    Args:
        cmap (List[Tuple[float, float, float]]): The [0, 255] RGB colormap to be checked.
    """
    cmap_lab = rgb_cmap_to_lab_cmap(cmap.copy())
    for i in range(len(cmap) - 1):
        # delta_e ambos colores
        delta_e_value_original = delta_e.deltaE_cie76(
            cmap_lab[i], cmap_lab[i + 1])
        print("delta_e izquierda y derecha :",
              round(delta_e_value_original, 2))
        print()
        # delta_e opuesto izq y derecho
        print(
            "delta_e izquierda y su opuesto :",
            round(delta_e.deltaE_cie76(cmap_lab[i],
                                       complemento_lab(cmap_lab[i])), 2),
        )
        print()
        # delta_e opuesto der y izq
        print(
            "delta_e derecha y su opuesto  :",
            round(
                delta_e.deltaE_cie76(cmap_lab[i + 1],
                                     complemento_lab(cmap_lab[i + 1])),
                2,
            ),
        )
        print()
        if delta_e_value_original > 12:
            print("representación correcta", "\n")
        else:
            print("representación incorrecta", "\n")


def binary_deviation(cmap, indices, cmap_result):
    """
    Display the binary deviation between two color maps.
    Args:
        cmap (list): The [0, 255] RGB color map.
        indices (list): The indices of the colors to consider in the binary deviation calculation.
        cmap_result (list): The resulting [0, 255] RGB color map.
    """

    binary_cmap = [cmap[i] for i in indices]
    binary_cmap_lab = rgb_cmap_to_lab_cmap(binary_cmap)
    binary_cmap_result_lab = rgb_cmap_to_lab_cmap(cmap_result)
    delta_e_list = []
    for i in range(len(indices)):
        delta_e_value = delta_e.deltaE_cie76(
            binary_cmap_lab[i], binary_cmap_result_lab[i]
        )
        delta_e_list.append(delta_e_value)
    print("desviación izquierda: ", delta_e_list[0])
    print("desviación derecha: ", delta_e_list[1])
