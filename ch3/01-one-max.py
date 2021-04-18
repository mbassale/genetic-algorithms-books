from deap import base
from deap import creator
from deap import tools
from pprint import pprint

import random
import matplotlib.pyplot as plt

ONE_MAX_LENGTH = 100  # length of binary string

# GenAlg parameters
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50

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

    # get all fitness values as array for plotting
    fitnessValues = [individual.fitness.values[0] for individual in population]

    maxFitnessValues = []
    meanFitnessValues = []

    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter += 1

        # tournament selection with size 3
        offspring = toolbox.select(population, len(population))

        # clone the individuals to prevent modifying the original population
        offspring = list(map(toolbox.clone, offspring))

        # apply crossover operator
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # apply mutation operator
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # recalculate fitness for new individuals
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        # replace with new population
        population[:] = offspring

        # get updated fitness array
        fitnessValues = [ind.fitness.values[0] for ind in population]

        # get statistics
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print(f'- Generation {generationCounter}: Max Fitness = {maxFitness}, Avg Fitness = {meanFitness}')

        best_index = fitnessValues.index(max(fitnessValues))
        print('Best individual = ', *population[best_index], '\n')

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()

if __name__ == '__main__':
    main()
