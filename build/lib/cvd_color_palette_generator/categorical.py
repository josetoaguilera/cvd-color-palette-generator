# from preprocess import *
# from aux_functions import *
from aux_functions import (
    get_color_difference_matrix,
    get_color_difference_matrix_v2,
    get_similar_colors_index,
    interpolate_colors_lab_to_lab,
    rgb_cmap_to_lab_cmap,
    split_universe,
)
import numpy as np
from skimage.color import delta_e


def get_better_cathegorical_group(splitted_universe, cmap_lab):
    """
    Calculates the better categorical group based on color difference metrics (mean and std).
    Args:
        splitted_universe (list(list)): A list of lists with indexes representing the subgroups of the splitted universe.
        cmap_lab (list): A color map in LAB format.
    Returns:
        data (list(list)): A sorted list of lists containing the mean and standard deviation of color difference matrix, along with the corresponding splitted universe.
    """

    data = []
    for s in splitted_universe:
        new_cmap_lab = []
        for i in s:
            new_cmap_lab.append(cmap_lab[i])
        matrix_difference = get_color_difference_matrix(new_cmap_lab)
        data.append([np.mean(matrix_difference), np.std(matrix_difference), s])
    data.sort(reverse=True)
    return data


def categorical_selection(cmap, min_value):
    """
    Selects the best categorical group from a colormap based on a minimum value.
    Args:
        cmap (list): The colormap to select from.
        min_value (float): The minimum value for color difference.
    Returns:
        result (list(int)): The selected categorical group indexes.
    """

    cmap_lab = rgb_cmap_to_lab_cmap(cmap)
    matrix_difference = get_color_difference_matrix(cmap_lab)
    less_than_n_list = get_similar_colors_index(matrix_difference, min_value)
    less_than_n_list = list(map(lambda x: [x[0], x[1] + x[0] + 1],
                                less_than_n_list))
    u_list = list(range(len(cmap)))
    split_index = less_than_n_list
    splitted_universe = split_universe(u_list, split_index)
    data = get_better_cathegorical_group(splitted_universe, cmap_lab)
    return data[0][2]


# cathegorical para cvd
def categorical_selection_cvd(cmap, cvd, min_value):
    """
    Selects the best categorical group of indexes from a given colormap considering color vision deficiency (CVD) and the value in which two colors are different.
    Args:
        cmap (lis(int)): The color map in rgb [0, 255].
        cvd (dict): A dictionary containing color maps for different types of color vision deficiency.
        min_value (float): The minimum color difference threshold.
    Returns:
        result (list(int)): The best group of color indexes for the categorical representation.
    """

    if len(cvd) == 0:
        result = categorical_selection(cmap, min_value)
        return result

    cmap_lab = rgb_cmap_to_lab_cmap(cmap)

    numpy_colors_cvd_list = {}
    for nombre in cvd:
        numpy_colors_cvd_list[nombre] = np.array(cvd[nombre])

    cmap_lab_cvd = {}
    for nombre in cvd.keys():
        cmap_lab_cvd[nombre] = rgb_cmap_to_lab_cmap(cvd[nombre])

    matrix_difference = get_color_difference_matrix_v2(cmap_lab)
    less_than_n_list = get_similar_colors_index(matrix_difference, min_value)
    less_than_n_list = list(map(lambda x: [x[0], x[1] + x[0] + 1],
                                less_than_n_list))
    matrix_difference_cvd_list = {}
    less_than_n_cvd_list = {}
    less_than_n_result = less_than_n_list.copy()

    for nombre in cmap_lab_cvd.keys():
        difference_cvd = get_color_difference_matrix_v2(cmap_lab_cvd[nombre])
        matrix_difference_cvd_list[nombre] = difference_cvd

        less_than_n_list = get_similar_colors_index(difference_cvd, min_value)
        less_than_n_list = list(
            map(lambda x: [x[0], x[1] + x[0] + 1], less_than_n_list)
        )

        for i in less_than_n_list:
            if i not in less_than_n_result:
                less_than_n_result.append(i)

        less_than_n_cvd_list[nombre] = less_than_n_list

    u_list = list(range(len(cmap)))
    split_index = less_than_n_result

    splitted_universe = split_universe(u_list, split_index)

    data_original = get_better_cathegorical_group(splitted_universe, cmap_lab)

    group_list_cvd = {}
    data_cvd = {}
    for nombre in cmap_lab_cvd.keys():
        group_list_cvd[nombre] = [
            g[2]
            for g in get_better_cathegorical_group(
                splitted_universe, cmap_lab_cvd[nombre]
            )
        ]
        data_cvd[nombre] = get_better_cathegorical_group(
            splitted_universe, cmap_lab_cvd[nombre]
        )

    data_index_list = []
    for g in data_original:
        cvd_sum = 0
        cvd_sum_std = 0
        for g_list_name in group_list_cvd:
            g_cvd_list = group_list_cvd[g_list_name]
            if (g[2] in g_cvd_list and cvd_sum is not None and
                    cvd_sum_std is not None):
                cvd_sum += g[0]
                cvd_sum_std += g[1]
            else:
                cvd_sum = None
        if cvd_sum is not None:
            cvd_sum /= len(group_list_cvd)
            cvd_sum_std /= len(group_list_cvd)
            data_mean_value = g[0] * 0.5 + cvd_sum * 0.5
            data_std_value = g[1] * 0.5 + cvd_sum_std * 0.5
            data_index_list.append([data_mean_value, data_std_value, g[2]])
    data_index_list = sorted(
        sorted(data_index_list, key=lambda a: a[1]),
        key=lambda a: a[0], reverse=True
    )
    return data_index_list[0][2]


def get_equal_luminance(cmap):
    """
    Transforms the given colormap to have equal luminance for all colors.
    Args:
        cmap (list(int)): The rgb [0, 255] colormap to be transformed.
    Returns:
        result (list(float)): The transformed colormap with equal luminance for all colors in lab color space.
    """

    colores_transformados = []
    def l_fun(a): return a / 255
    for i in cmap:
        mapped = list(map(l_fun, i))
        colores_transformados.append(mapped)

    # para cada color en rgb transformarlos a lab
    lab_cmap = rgb_cmap_to_lab_cmap(cmap)

    # obtener luminosidad promedio de los colores
    average_luminance = np.mean([c[0] for c in lab_cmap])

    # transformar todos los colores a la luminosidad promedio
    transformed_lab_cmap = [(average_luminance, c[1], c[2]) for c in lab_cmap]

    return transformed_lab_cmap


def categorical_representation_selected(cmap, luminosity_equality):
    """
    Generates a categorical representation of a color map.
    Args:
        cmap (list): The rgb [0, 255] color map to work with.
        luminosity_equality (int): The desired luminosity equality.
    Returns:
        result (numpy.ndarray): The categorical representation of the selected color map in Lab color space.
    """

    # color a trabajar
    cmap_work = cmap.copy()

    equal_l_cmap = get_equal_luminance(cmap_work)

    color_map_categoricos = []

    for j in range(len(equal_l_cmap)):
        cmap_work_lab = rgb_cmap_to_lab_cmap(cmap_work)
        two_colors = [cmap_work_lab[j], equal_l_cmap[j]]
        interpolated_colors = interpolate_colors_lab_to_lab(two_colors,
                                                            n_colors=100)

        color_map_categoricos.append(
            interpolated_colors[luminosity_equality - 1])
    return np.array(color_map_categoricos)


# chequeo categorico
# diferenciabilidad
# delta_e entre todos los colores similares, y mayor a 11
# luminosidad similar entre todos los colores
def check_categorical(cmap):
    """
    Check if the given [0, 255] rgb colormap represents categorical values.
    Args:
        cmap (List[Tuple[float, float, float]]): The [0, 255] RGB colormap to be checked.
    """
    # matriz de diferenciabilidad
    cmap_lab = rgb_cmap_to_lab_cmap(cmap.copy())
    delta_e_matrix = get_color_difference_matrix_v2(cmap_lab)

    delta_e_check_value = True

    print("Matriz delta_e: ")
    for i in range(len(delta_e_matrix)):
        if delta_e_matrix[i] != []:
            delta_e_value = []
            for j in range(len(delta_e_matrix[i])):
                delta_e_value.append(round(delta_e_matrix[i][j], 2))
                if delta_e_matrix[i][j] < 12:
                    delta_e_check_value = False

            delta_e_value = [round(d, 2) for d in delta_e_matrix[i]]
            print("color #", i + 1, " : ", delta_e_value, "\n")
    for j in range(len(cmap)):
        luminosity_value = cmap_lab[j][0]
        print("lumminosidad de color #", j + 1, ": ",
              round(luminosity_value, 2))
    print()
    (
        print("diferenciabilidad correcta")
        if delta_e_check_value
        else print("diferenciabilidad incorrecta")
    )


def categorical_deviation(cmap, indices, cmap_result):
    """
    Display the categorical deviation between the colors in the colormap.
    Args:
        cmap (list): The [0, 255] RGB color map.
        indices (list): The indices of the colors to consider in the categorical deviation calculation.
        cmap_result (list): The resulting [0, 255] RGB color map.
    """
    categorical_cmap = [cmap[i] for i in indices]
    categorical_cmap_lab = rgb_cmap_to_lab_cmap(categorical_cmap)
    categorical_cmap_result_lab = rgb_cmap_to_lab_cmap(cmap_result)
    delta_e_list = []
    for i in range(len(indices)):
        delta_e_value = delta_e.deltaE_cie76(
            categorical_cmap_lab[i], categorical_cmap_result_lab[i]
        )
        delta_e_list.append(delta_e_value)
        print("desviación ", i, ": ", delta_e_value)
