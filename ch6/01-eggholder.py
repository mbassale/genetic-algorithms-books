from deap import base
from deap import creator
from deap import tools

import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import elitism

# problem constants
DIMENSIONS = 2
BOUND_LOW, BOUND_HIGH = -512.0, 512.0

# genetic algorithm constants
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.25
MAX_GENERATIONS = 300
HALL_OF_FAME_SIZE = 30
# crowding factor for crossover and mutation
CROWDING_FACTOR = 20.0

# set the random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

# create the Individual class based on list
creator.create('Individual', list, fitness=creator.FitnessMin)


# helper function for creating random real numbers uniformly distributed within a given range [low, high]
# it assumes that the range is the same for every dimension
def randomFloat(low, high):
    return [random.uniform(l, h) for l, h in zip([low] * DIMENSIONS, [high] * DIMENSIONS)]


# create an operator that randomly returns a float in the desired range and dimension
toolbox.register('attrFloat', randomFloat, BOUND_LOW, BOUND_HIGH)

# create the individual operator to fill up an Individual instance
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.attrFloat)

# create the population operator to generate a list of individuals
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


# Eggholder function as the given individual's fitness
def eggholder(individual):
    x = individual[0]
    y = individual[1]
    f = (-(y + 47.0) * np.sin(np.sqrt(abs(x / 2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
    return f,  # return a tuple


toolbox.register('evaluate', eggholder)

# genetic operators
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_HIGH, eta=CROWDING_FACTOR)
toolbox.register('mutate', tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_HIGH, eta=CROWDING_FACTOR,
                 indpb=1.0 / DIMENSIONS)


# Genetic Algorithm Flow
def main():
    # create initial population
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)

    # define the hall-of-fame object
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with elitism
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print info for best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    plt.show()


if __name__ == "__main__":
    main()
