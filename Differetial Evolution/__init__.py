from Point import Point
from Population import Population
from test_functions import Function


def get_best_point(points):
    best = sorted(points, key=lambda x:x.z)[0]
    return best


if __name__ == '__main__':
	pass