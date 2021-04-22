import array

import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import seaborn as sns
from vrp import VRP
from elitism import eaSimpleWithElitism

TSP_NAME = 'bayg29'
NUM_OF_VEHICLES = 6
DEPOT_LOCATION = 12
vrp = VRP(TSP_NAME, NUM_OF_VEHICLES, DEPOT_LOCATION)

# GenAlg parameters
POPULATION_SIZE = 500
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 1000
HALL_OF_FAME_SIZE = 30

# Fix random generator parameters
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Register function to generate random bits
toolbox = base.Toolbox()
toolbox.register('zeroOrOne', random.randint, 0, 1)

# Register fitness strategy
creator.create('FitnessMin', base.Fitness, weights=(-1.0, 1.0, -1.0))

# Create individual class
creator.create('Individual', array.array, typecode='i', fitness=creator.FitnessMin)

# Create operator to shuffle the cities
toolbox.register('randomOrder', random.sample, range(len(vrp)), len(vrp))

# Create initial random individual operator
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.randomOrder)

# Create random population operator
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


def vrpFitness(individual) -> tuple:
    return vrp.getMaxDistance(individual), vrp.getMinDistance(individual), vrp.getAvgDistance(individual)


toolbox.register('evaluate', vrpFitness)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxUniformPartialyMatched, indpb=2.0 / len(vrp))
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0 / len(vrp))


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)

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
    vrp.plotData(best)

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
