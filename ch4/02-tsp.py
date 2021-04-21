import array

import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import seaborn as sns
from tsp import TSP
from elitism import eaSimpleWithElitism

TSP_NAME = 'bayg29'
tsp = TSP(TSP_NAME)

# GenAlg parameters
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 30

# Fix random generator parameters
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Register function to generate random bits
toolbox = base.Toolbox()
toolbox.register('zeroOrOne', random.randint, 0, 1)

# Register fitness strategy
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

# Create individual class
creator.create('Individual', array.array, typecode='i', fitness=creator.FitnessMin)

# Create operator to shuffle the cities
toolbox.register('randomOrder', random.sample, range(len(tsp)), len(tsp))

# Create initial random individual operator
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.randomOrder)

# Create random population operator
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


def tspFitness(individual) -> tuple:
    return tsp.getTotalDistance(individual),


toolbox.register('evaluate', tspFitness)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0 / len(tsp))


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # evaluate initial fitness and store on individual
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best = hof.items[0]
    print('Best Individual = ', best)
    print('Best Fitness = ', best.fitness.values[0])
    plt.figure(1)
    tsp.plotData(best)

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()


if __name__ == '__main__':
    main()
