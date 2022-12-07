import math
import matplotlib.pyplot as plt
import numpy as np

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=30, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

def fmodel(x,w):
    return 10 + (w[0] * (x**2)) - (10*np.cos(w[1]*x))

def rmse(w):
    y_pred = fmodel(x, w)
    return np.sqrt(sum((y - y_pred)**2) / len(y))

if __name__ == '__main__':

    # Generate data set
    x = np.linspace(-5.12, 5.12, 500)
    y = (10 + x**2 - 10*np.cos(2*math.pi*x)) + np.random.normal(0, 0.2, 500)
    #USe GA to get results
    result = list(de(rmse, [(-5.12, 5.12)]*2 , its=1000))
    print(result)
    plt.scatter(x, y)
    plt.plot(x, (10 + x**2 - 10*np.cos(2*math.pi*x)), label='Rastrigin')
    plt.plot(x, fmodel(x, [0.4886555 , 0.79301061]), label = 'result')
    plt.legend()
    plt.show()