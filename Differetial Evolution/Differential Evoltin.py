import numpy as np
import pandas as pd
from functions import rastrigin
import random

def rastrigin_error(x, dim):
  # f(X) = Sum[xj^2 – 10*cos(2*pi*xj)] + 10n
  z = 0.0
  for j in range(dim):
    z += x[j]**2 - (10.0 * np.cos(2*np.pi*x[j]))
  z += dim * 10.0
  # return avg squared difference from true min at 0.0
  err = (z - 0.0)**2
  return err


def run(dim,population,Fr,Cr,generations,bound):
    print("\nBegin Differential Evolution optimization demo ")
    print("Goal is to minimize Rastrigin function with dim = 3 ")
    print("Function has known min = 0.0 at (0, 0, 0) ")
    np.random.seed(1)
    np.set_printoptions(precision=5, suppress=True, sign=" ")

    dim = dim
    pop_size = population
    F = Fr  # mutation
    C = Cr  # crossover
    max_gen = generations
    hbound=bound
    lbound=-1*bound
    goal=0
    print("\nSetting pop_size = %d, F = %0.1f, cr = %0.2f, \
max_gen = %d " % (pop_size, F, C, max_gen))

    # create array-of-arrays population of random solutions
    print("\nCreating random solutions and computing their error ")
    population = np.random.uniform(lbound, hbound, (pop_size, dim))
    popln_errors = np.zeros(pop_size)
    for i in range(pop_size):
        popln_errors[i] = rastrigin_error(population[i], dim)
    # find initial best solution and its error
    # best_idx = np.argmin(popln_errors)
    # best_error = popln_errors[best_idx]

    # main processing loop
    t=0
    for g in range(max_gen):
        t=g
        for i in range(pop_size):  # each possible soln in pop
            # pick 3 other possible solns
            indices = np.arange(pop_size)  # [0, 1, 2, . . ]
            np.random.shuffle(indices)
            for j in range(3):
                if indices[j] == i: indices[j] = indices[pop_size - 1]

            # use the 3 others to create a mutation
            a = indices[0]
            b = indices[1]
            c = indices[2]
            mutation = population[a] + F * (population[b] \
                                            - population[c])
            for k in range(dim):
                if mutation[k] != bound: mutation[k] = bound

            # use the mutation and curr item to create a
            # new solution item
            new_soln = np.zeros(dim)
            for k in range(dim):
                p = np.random.random()  # between 0.0 and 1.0
                if p < c:  # usually
                    new_soln[k] = mutation[k]
                else:
                    new_soln[k] = population[i][k]  # use current item

            # replace curr soln with new soln if new soln is better
            new_soln_err = rastrigin_error(new_soln, dim)
            if new_soln_err < popln_errors[i]:
                population[i] = new_soln
                popln_errors[i] = new_soln_err

        # after all popln items have been processed,
        # find curr best soln
        best_idx = np.argmin(popln_errors)
        best_error = popln_errors[best_idx]
        if g % 10 == 0:
            print("Generation = %4d best error = %10.4f \
  best_soln = " % (g, best_error), end="")
            print(population[best_idx])

    # show final result
    best_idx = np.argmin(popln_errors)
    best_error = popln_errors[best_idx]
    print("\nFinal best error = %0.4f  best_soln = " % \
          best_error, end="")
    print(population[best_idx])

    print("\nEnd demo ")

    if best_error<1e-08:
        goal=1

    return(t,goal,best_error)

#######




def initialization(n, mu, max):
    individuals = np.random.uniform(low=-1 * max, high=max, size=(mu, n))
    return individuals


def crossover(individuals, mutants):
    trials = np.zeros_like(individuals)
    for g in range(individuals.shape[0]):
        for i in range(individuals.shape[1]):
            random_parent = random.normalvariate(0, 1)
            if random_parent > 0.5:
                trials[g, i] = individuals[g, i]
            else:
                trials[g, i] = mutants[g, i]
        random_position = random.randint(0, individuals.shape[1] - 1)
        trials[g, random_position] = individuals[g, i]
    return trials


def make_mutants(individuals):
    mutants = np.zeros_like(individuals)
    for i in range(individuals.shape[0]):
        choosed = individuals[np.random.choice(individuals.shape[0], 3, replace=False), :]
        mutants[i] = choosed[0] + f * (choosed[1] - choosed[2])
    return mutants


def execute(individuals,dim,Gmax):
    gen = 0
    gens = []
    goal=0
    best_mins = []
    while gen < Gmax:
        gen = gen + 1
        mutants = make_mutants(individuals=individuals)
        trials = crossover(individuals, mutants)

        individuals_fitness = np.apply_along_axis(function, axis=1, arr=individuals)
        trials_fitness = np.apply_along_axis(function, axis=1, arr=trials)

        next_generation = np.zeros_like(individuals)
        for i in range(individuals.shape[1]):
            if individuals_fitness[i] < trials_fitness[i]:
                next_generation[i] = individuals[i]
            else:
                next_generation[i] = trials[i]

        individuals = next_generation
        best_mins.append(np.mean(np.apply_along_axis(function, axis=1, arr=individuals)))
        print(f'Generation = {gen}')
        gens.append(gen)
    if best_mins[gen-1]<1e-08:
        goal=1

    return(gens[gen-1],goal,best_mins[gen-1])

if __name__ == "__main__":
    for i in [30,60,90]:
        d = 3
        N = i
        Frate=0.5
        Crate=0.7
        Gmax = 1000
        bound = 5.12
        T = list(range(1, 200))
        success = []
        best_err = []
        iterations = []

        for k in T:
            t, goal, err = run(d,i,Frate,Crate,Gmax,bound)

            iterations.append(t)
            success.append(goal)
            best_err.append(err)


    Evalaution = pd.DataFrame({'Evalaution Turn': T,
                                       'iterations': iterations,
                                       'success': success,
                                       'Error': best_err

                                       })
    Evalaution.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/DE Results/Rastrigin"+str(i)+".csv")

    '''for i in [30, 60, 90]:
        function=rastrigin
        n = i
        f = 0.9
        max_range = 5.12
        mu = 100
        d=3
        individuals = initialization(n=n, mu=mu, max=max_range)
        Gmax=100
        T = list(range(1, 10))
        success = []
        best_err = []
        iterations = []

        for k in T:
            t, goal, err = execute(individuals,d,Gmax)

            iterations.append(t)
            success.append(goal)
            best_err.append(err)

        Evalaution = pd.DataFrame({'Evalaution Turn': T,
                               'iterations': iterations,
                               'success': success,
                               'Error': best_err

                               })
        Evalaution.to_csv("/Users/ahmed/Documents/UOE/Courses/Semester 1/Natural COmputing /Project 40%/Particle_Swarm_Optimization-main/Rastrigin_"+str(i)+".csv")'''