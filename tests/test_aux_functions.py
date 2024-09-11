import pytest
import numpy as np
from colormath.color_objects import LabColor
from cvd_color_palette_generator.aux_functions import (
    get_max_min_indices,
    rgb_cmap_to_lab_cmap,
    lab_cmap_to_rgb_cmap,
    rgb_to_lab,
    lab_to_rgb,
    interpolate_colors_lab_to_lab,
    complemento_lab,
    get_indexes_below_n,
    get_color_difference_matrix,
    get_color_difference_matrix_v2,
    find_max_index_in_list_of_lists,
    split_universe,
    obtener_posiciones_ordenadas,
    delta_l_matrix,
    filter_tuples_by_exact_numbers
)

def test_get_max_min_indices():
    lst = [[3, 1, 4], [1, 2, 3], [4, 0, 2], [2, 0, 1]]
    assert get_max_min_indices(lst) == (2, 1)

    lst = []
    assert get_max_min_indices(lst) == (None, None)

def test_rgb_cmap_to_lab_cmap():
    cmap = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    lab_cmap = rgb_cmap_to_lab_cmap(cmap)
    assert len(lab_cmap) == 3  # Checks if conversion returns 3 lab values.
    result = [[53.23, 80.11, 67.22], [87.74, -86.18, 83.18], [32.30, 79.20, -107.86]]
    iguales = np.isclose(lab_cmap, result, rtol=0.01, atol=0.3)
    assert iguales.all()

def test_lab_cmap_to_rgb_cmap():
    cmap_lab = [[53.23, 80.11, 67.22], [87.74, -86.18, 83.18]]
    rgb_cmap = lab_cmap_to_rgb_cmap(cmap_lab)
    assert len(rgb_cmap) == 2
    result = [[255, 0, 0], [0, 255, 0]]
    rgb_cmap = np.multiply(rgb_cmap, 255)
    iguales = np.isclose(rgb_cmap, result, rtol=0.01, atol=0.3)
    assert iguales.all()

def test_rgb_to_lab():
    rgb = (255, 0, 0)
    lab = rgb_to_lab(rgb)
    assert isinstance(lab, LabColor)
    assert 0 <= lab.lab_l <= 100
    assert -128 <= lab.lab_a <= 128
    assert -128 <= lab.lab_b <= 128
    assert lab.lab_l == pytest.approx(53.23, 0.01)
    assert lab.lab_a == pytest.approx(80.11, 0.01)
    assert lab.lab_b == pytest.approx(67.22, 0.01)

def test_lab_to_rgb():
    lab = LabColor(53.23288178584245, 80.10930952982204, 67.22006831026425)
    rgb = lab_to_rgb(lab)
    assert all(0 <= channel <= 1 for channel in rgb)
    assert (1, 0, 0) == (round(rgb[0], 1), round(rgb[1], 1), round(rgb[2], 1))

def test_interpolate_colors_lab_to_lab():
    colors = [(50, 0, 0), (75, -25, 25)]
    n_colors = 5
    interpolated = interpolate_colors_lab_to_lab(colors, n_colors=n_colors)
    assert len(interpolated) == n_colors
    iguales = np.isclose(interpolated[0], [50, 0, 0], rtol=0.01, atol=0.3)
    assert iguales.all()
    iguales = np.isclose(interpolated[-1], [75, -25, 25], rtol=0.01, atol=0.3)
    assert iguales.all()
    iguales = np.isclose(interpolated[2], [62.5, -12.5, 12.5], rtol=0.01, atol=0.3)
    assert iguales.all()

def test_complemento_lab():
    lab_color = [50, 25, -25]
    complement = complemento_lab(lab_color)
    assert complement == [50, -25, 25]

def test_get_indexes_below_n():
    matrix = [[1, 2], [3, 4]]
    n = 3
    indexes = get_indexes_below_n(matrix, n)
    assert indexes == [(0, 0), (0, 1)]
    matrix = [
       [  0.  ,   3.47,   5.26,  48.58,  87.21,  18.35,  28.32,  54.1 , 89.11,  99.64,  75.17,  84.69,  86.27, 101.16,  50.67,  92.35, 50.12, 113.14,  39.77, 103.74],
       [  3.47,   0.  ,   1.84,  46.62,  85.37,  15.79,  24.86,  50.63, 85.64,  96.17,  72.92,  83.89,  85.69,  98.67,  48.96,  90.59, 48.43, 111.75,  38.45, 102.33],
       [  5.26,   1.84,   0.  ,  46.1 ,  84.91,  14.97,  23.06,  48.86, 83.87,  94.41,  71.45,  83.09,  84.97,  97.78,  48.08,  90.19, 47.92, 110.92,  37.49, 101.49],
       [ 48.58,  46.62,  46.1 ,   0.  ,  42.99,  31.42,  40.8 ,  47.75, 72.82,  80.99,  97.9 , 121.22, 124.59,  59.97,  52.16,  48.12, 22.93, 133.3 ,  75.43, 124.63],
       [ 87.21,  85.37,  84.91,  42.99,   0.  ,  71.42,  77.13,  74.48, 84.62,  89.28, 126.85, 152.82, 158.07,  28.3 ,  92.45,   5.86, 61.36, 145.92, 108.52, 138.82],
       [ 18.35,  15.79,  14.97,  31.42,  71.42,   0.  ,  18.92,  41.61, 75.8 ,  85.97,  76.4 ,  93.63,  96.08,  84.74,  42.55,  76.77, 34.79, 116.44,  47.43, 107.14],
       [ 28.32,  24.86,  23.06,  40.8 ,  77.13,  18.92,   0.  ,  25.82, 60.85,  71.4 ,  58.92,  80.84,  84.08,  84.24,  43.08,  82.89, 44.1 , 104.13,  36.71,  94.82],
       [ 54.1 ,  50.63,  48.86,  47.75,  74.48,  41.61,  25.82,   0.  , 35.04,  45.58,  55.44,  87.31,  91.77,  73.19,  51.24,  80.28, 52.15, 102.87,  51.33,  94.28],
       [ 89.11,  85.64,  83.87,  72.82,  84.62,  75.8 ,  60.85,  35.04, 0.  ,  10.72,  67.66, 105.29, 110.74,  71.58,  77.03,  89.61, 77.57, 109.76,  80.21, 102.95],
       [ 99.64,  96.17,  94.41,  80.99,  89.28,  85.97,  71.4 ,  45.58, 10.72,   0.  ,  75.53, 113.62, 119.11,  73.68,  85.14,  93.91, 85.43, 115.87,  90.57, 109.61],
       [ 75.17,  72.92,  71.45,  97.9 , 126.85,  76.4 ,  58.92,  55.44, 67.66,  75.53,   0.  ,  38.68,  43.9 , 123.37,  82.72, 132.63, 99.28,  75.94,  40.42,  68.04],
       [ 84.69,  83.89,  83.09, 121.22, 152.82,  93.63,  80.84,  87.31, 105.29, 113.62,  38.68,   0.  ,  10.24, 153.11, 104.81, 158.5 , 122.6 ,  73.12,  47.05,  66.28],
       [ 86.27,  85.69,  84.97, 124.59, 158.07,  96.08,  84.08,  91.77, 110.74, 119.11,  43.9 ,  10.24,   0.  , 159.43, 104.33, 163.79, 124.25,  83.11,  50.94,  76.36],
       [101.16,  98.67,  97.78,  59.97,  28.3 ,  84.74,  84.24,  73.19, 71.58,  73.68, 123.37, 153.11, 159.43,   0.  , 103.15,  29.52, 77.42, 138.46, 113.25, 132.33],
       [ 50.67,  48.96,  48.08,  52.16,  92.45,  42.55,  43.08,  51.24, 77.03,  85.14,  82.72, 104.81, 104.33, 103.15,   0.  ,  97.53, 34.04, 143.89,  69.3 , 134.54],
       [ 92.35,  90.59,  90.19,  48.12,   5.86,  76.77,  82.89,  80.28, 89.61,  93.91, 132.63, 158.5 , 163.79,  29.52,  97.53,   0.  ,65.95, 150.44, 114.13, 143.49],
       [ 50.12,  48.43,  47.92,  22.93,  61.36,  34.79,  44.1 ,  52.15, 77.57,  85.43,  99.28, 122.6 , 124.25,  77.42,  34.04,  65.95, 0.  , 146.14,  79.03, 137.06],
       [113.14, 111.75, 110.92, 133.3 , 145.92, 116.44, 104.13, 102.87, 109.76, 115.87,  75.94,  73.12,  83.11, 138.46, 143.89, 150.44, 146.14,   0.  ,  80.56,   9.53],
       [ 39.77,  38.45,  37.49,  75.43, 108.52,  47.43,  36.71,  51.33, 80.21,  90.57,  40.42,  47.05,  50.94, 113.25,  69.3 , 114.13, 79.03,  80.56,   0.  ,  71.1 ],
       [103.74, 102.33, 101.49, 124.63, 138.82, 107.14,  94.82,  94.28, 102.95, 109.61,  68.04,  66.28,  76.36, 132.33, 134.54, 143.49, 137.06,   9.53,  71.1 ,   0.  ]]
    indexes = get_indexes_below_n(matrix, 12)
    assert indexes == [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,3),(4,4),(4,15),(5,5),(6,6),(7,7),(8,8),(8,9),(9,8),(9,9),(10,10),(11,11),(11,12),(12,11),(12,12),(13,13),(14,14),(15,4),(15,15),(16,16),(17,17),(17,19),(18,18),(19,17),(19,19)]

def test_get_color_difference_matrix():
    cmap = [[50, 0, 0], [75, -25, 25]]
    matrix = get_color_difference_matrix(cmap)
    assert len(matrix) == 2  # Ensure matrix has the correct size.
    assert matrix[0][0] == 0  # Ensure diagonal is 0.
    assert matrix[0][1] == pytest.approx(43.30, 0.01)
    assert matrix[1][0] == pytest.approx(43.30, 0.01)
    assert matrix[1][1] == 0

def test_find_max_index_in_list_of_lists():
    lst = [[1, 3], [2, 4]]
    max_index = find_max_index_in_list_of_lists(lst)
    assert max_index == (1, 1)
    lst = [[1, 3], [7, 4], [5, 6]]
    max_index = find_max_index_in_list_of_lists(lst)
    assert max_index == (1, 0)

def test_split_universe():
    u_list = [1, 2, 3, 4, 5]
    split_index = [[1, 3], [2, 4]]
    result = split_universe(u_list, split_index)
    assert len(result) > 0
    np.testing.assert_array_equal(result, [[3, 4, 5], [2, 3, 5], [1, 4, 5], [1, 2, 5]])

def test_obtener_posiciones_ordenadas():
    matrix = [[10, 20], [30, 40]]
    n = 15
    positions = obtener_posiciones_ordenadas(matrix, n)
    assert positions == [(1, 1), (1, 0), (0, 1)]
    matrix = [
       [  0.  ,   3.47,   5.26,  48.58,  87.21,  18.35,  28.32,  54.1 , 89.11,  99.64  ],
       [  3.47,   0.  ,   1.84,  46.62,  85.37,  15.79,  24.86,  50.63, 85.64,  96.17  ],
       [  60.85,  35.04, 0.  ,  10.72,  67.66, 105.29, 110.74,  71.58,  77.03,  89.61  ],
       [  71.4 ,  45.58, 10.72,   0.  ,  75.53, 113.62, 119.11,  73.68,  85.14,  93.91 ],
       [  87.31, 105.29, 113.62,  38.68,   0.  ,  10.24, 153.11, 104.81, 158.5 , 122.6 ],
       [  91.77, 110.74, 119.11,  43.9 ,  10.24,   0.  , 159.43, 104.33, 163.79, 124.25],
       [  93.91, 132.63, 158.5 , 163.79,  29.52,  97.53,   0.  ,65.95, 150.44, 114.13  ],
       [  85.43,  99.28, 122.6 , 124.25,  77.42,  34.04,  65.95, 0.  , 146.14,  79.03  ],
       [  40.42,  47.05,  50.94, 113.25,  69.3 , 114.13, 79.03,  80.56,   0.  ,  71.1  ],
       [  68.04,  66.28,  76.36, 132.33, 134.54, 143.49, 137.06,   9.53,  71.1 ,   0.  ]]
    n = 12
    positions = obtener_posiciones_ordenadas(matrix, n)
    assert sorted(positions) == sorted([(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(2,0),(2,1),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(4,0),(4,1),(4,2),(4,3),(4,6),(4,7),(4,8),(4,9),(5,0),(5,1),(5,2),(5,3),(5,6),(5,7),(5,8),(5,9),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,7),(6,8),(6,9),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,8),(7,9),(8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,6),(8,7),(8,9),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,8)])

def test_delta_l_matrix():
    cmap_lab = [[50, 0, 0], [75, -25, 25]]
    matrix = delta_l_matrix(cmap_lab)
    assert len(matrix) == 2  # Check if matrix size is correct.
    assert matrix[0][0] == 0  # Check if diagonal is 0.
    assert matrix[0][1] == -25  # Check if value is correct.
    assert matrix[1][0] == 25  # Check if value is correct.
    assert matrix[1][1] == 0  # Check if value is correct.

def test_filter_tuples_by_exact_numbers():
    tuple_array = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    number_array = [2, 3, 5, 4, 6, 7]
    filtered = filter_tuples_by_exact_numbers(tuple_array, number_array)
    assert filtered == [(4, 5, 6)]