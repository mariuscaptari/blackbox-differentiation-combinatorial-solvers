import numpy as np
import pytest
from comb_modules.a_star import a_star, manhattan_heuristic

@pytest.fixture
def simple_matrix():
    return np.array([
        [1, 1, 9, 1],
        [1, 9, 9, 1],
        [1, 1, 1, 1]
        ], dtype=np.float32)

@pytest.fixture
def complex_matrix():
    return np.array([
        [1, 1, 4, 1, 1],
        [1, 2, 3, 1, 3],
        [1, 2, 2, 1, 1],
        [1, 4, 2, 1, 1],
        ], dtype=np.float32)

def test_a_star_simple_path(simple_matrix):
    # Test A* on a simple grid with no obstacles
    expected_path = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 1]
    ], dtype=np.float32)
    result = a_star(simple_matrix).shortest_path
    np.testing.assert_array_equal(result, expected_path)

def test_a_start_complex_path(complex_matrix):
    # Test A* on a simple grid with no obstacles
    expected_path = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        ], dtype=np.float32)
    result = a_star(complex_matrix).shortest_path
    np.testing.assert_array_equal(result, expected_path)

def test_heuristic():
    # Test heuristic calculation
    start = (0, 0)
    goal = (2, 2)
    result = manhattan_heuristic(start, goal)
    assert result == 4

def test_a_star_unique_path():
    # Test if path uniqueness is correctly identified
    matrix = np.array([
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ], dtype=np.float32)
    result = a_star(matrix)
    assert result.is_unique

# def test_no_path_found():
#     # Test behavior when no path exists
#     matrix = np.array([
#         [1,   1,   1],
#         [np.inf, np.inf, np.inf],
#         [1,   1,   1]
#     ], dtype=np.float32)
#     result = a_star(matrix)
#     # Assuming function returns a shortest_path of all zeros when no path is found
#     assert not np.any(result.shortest_path)