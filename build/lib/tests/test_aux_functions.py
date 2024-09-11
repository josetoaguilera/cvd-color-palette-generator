import unittest
from cvd_color_palette_generator.aux_functions import obtener_posiciones_ordenadas, get_indexes_below_n, filter_tuples_by_exact_numbers, get_color_difference_matrix_v2, get_similar_colors_index


def test_obtener_posiciones_ordenadas():
    # Ejemplo de uso
    matrix = [
        [10, 25, 3],
        [22, 15, 12],
        [7, 18, 9]
    ]
    n = 10

    posiciones_ordenadas = obtener_posiciones_ordenadas(matrix, n)
    print(posiciones_ordenadas)

def test_filter_tuples_by_exact_numbers():
    # Ejemplo de uso
    tuple_array = [(1, 2), (3, 4), (5, 6), (7, 8)]
    number_array = [2, 5, 6, 9]

    filtered_tuples = filter_tuples_by_exact_numbers(tuple_array, number_array)
    print(filtered_tuples)

def test_get_color_difference_matrix_v2():
    # Ejemplo de uso
    cmap_lab = [[ 9.96549222e+01, -2.44763489e-03,  4.63957812e-03],
                [92.27206438,  1.1662534 ,  0.96515618],
                [81.49855428, -2.34881037, -1.93744156],
                [ 65.15591575,  -2.80226552, -28.12601194],
                [ 46.42197423,   3.99738218, -44.08835454],
                [70.6270355 , -2.33320452,  0.9252845 ],
                [ 5.74777564e+01, -1.55502867e-03,  2.94761160e-03],
                [43.54527376,  1.78600902, -1.26824611],
                [ 2.92888178e+01, -9.58458911e-04,  1.81679261e-03],
                [ 1.22500301e+01, -5.97862659e-04,  1.13326972e-03],
                [0., 0., 0.],
                [82.28183665, 21.59423532,  6.72914125],
                [70.3158071 , 39.04169978, 14.23474776],
                [59.28833052, 56.47282008, 24.50728085],
                [53.49510174, 67.85652775, 38.50502995],
                [ 67.30611704, -40.6413757 ,  19.12092634],
                [ 54.39052612, -51.22865418,  32.25540519],
                [93.98673656, -6.3104704 , 41.4975618 ],
                [89.73570731, -6.72139768, 74.71134388],
                [86.65264608, -3.20033276, 86.80697334]
                ]
    color_diff_matrix = get_color_difference_matrix_v2(cmap_lab)
    print(color_diff_matrix)

def test_get_similar_colors_index():
    cmap_lab = [[ 9.96549222e+01, -2.44763489e-03,  4.63957812e-03],
                [92.27206438,  1.1662534 ,  0.96515618],
                [81.49855428, -2.34881037, -1.93744156],
                [ 65.15591575,  -2.80226552, -28.12601194],
                [ 46.42197423,   3.99738218, -44.08835454],
                [70.6270355 , -2.33320452,  0.9252845 ],
                [ 5.74777564e+01, -1.55502867e-03,  2.94761160e-03],
                [43.54527376,  1.78600902, -1.26824611],
                [ 2.92888178e+01, -9.58458911e-04,  1.81679261e-03],
                [ 1.22500301e+01, -5.97862659e-04,  1.13326972e-03],
                [0., 0., 0.],
                [82.28183665, 21.59423532,  6.72914125],
                [70.3158071 , 39.04169978, 14.23474776],
                [59.28833052, 56.47282008, 24.50728085],
                [53.49510174, 67.85652775, 38.50502995],
                [ 67.30611704, -40.6413757 ,  19.12092634],
                [ 54.39052612, -51.22865418,  32.25540519],
                [93.98673656, -6.3104704 , 41.4975618 ],
                [89.73570731, -6.72139768, 74.71134388],
                [86.65264608, -3.20033276, 86.80697334]
                ]
    color_diff_matrix = get_color_difference_matrix_v2(cmap_lab)
    similar_color_index = get_similar_colors_index(color_diff_matrix, min_value=13)
    index_below_n = get_indexes_below_n(similar_color_index, 12)
    print(similar_color_index)
    print(index_below_n)
    print([[x, y] for x, y in index_below_n if x != y])

def test_get_indexes_below_n():

    get_indexes_below_n()

if __name__ == '__main__':
    unittest.main()