from Differential_Evloution  import DifferentialEvolution
import datetime

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
	number_of_runs = 1000
	val = 0
	print_time = True

	for i in range(number_of_runs):
		start = datetime.datetime.now()
		de = DifferentialEvolution(num_iterations=100, dim=3, CR=0.7, F=0.5, population_size=90, print_status=False, visualize=False,func='rastrigin')
		val += de.simulate()
		if print_time:
			print("\nTime taken:", datetime.datetime.now() - start)
	print('-'*80)
	print("\nFinal average of all runs:", val / number_of_runs)