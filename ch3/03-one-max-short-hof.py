from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from pprint import pprint
import numpy

import random
import matplotlib.pyplot as plt

ONE_MAX_LENGTH = 100  # length of binary string

# GenAlg parameters
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 10

# Fix random generator parameters
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Register function to generate random bits
toolbox = base.Toolbox()
toolbox.register('zeroOrOne', random.randint, 0, 1)

# Register fitness strategy
creator.create('FitnessMax', base.Fitness, weights=(1.0,))

# Create individual class
creator.create('Individual', list, fitness=creator.FitnessMax)

# Create initial random individual operator
toolbox.register('individualCreator', tools.initRepeat,
                 creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# Create random population operator
toolbox.register('populationCreator', tools.initRepeat,
                 list, toolbox.individualCreator)


# Fitness function


def oneMaxFitness(individual) -> tuple:
    return sum(individual),


# Register fitness function
toolbox.register("evaluate", oneMaxFitness)

# Create select, crossover and mutation operators
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)


# Genetic Flow


def main():
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    # evaluate initial fitness and store on individual
    fitnessValues = list(map(toolbox.evaluate, population))
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', numpy.max)
    stats.register('avg', numpy.mean)

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print('Best Individual = ', hof.items[0])

    maxFitnessValues, meanFitnessValues = logbook.select('max', 'avg')
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()
