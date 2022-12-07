import numpy as np




class Function:
    def __init__(self, func="rastrigin"):

        self.objectives = {
            'rastrigin': self.rastrigin,
        }


        self.func_name = func
        self.func = self.objectives[self.func_name]


    def evaluate(self, point):
        return self.func(point)


    def rastrigin(self, x):
        v = 0

        for i in range(len(x)):
            v += (x[i] ** 2) - (10 * np.cos(2 * np.pi * x[i]))

        return (10 * len(x)) + v


if __name__ == '__main__':
    print("A collection of several Test functions for optimizations")