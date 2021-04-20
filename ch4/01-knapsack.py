import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt


class Knapsack:
    def __init__(self, items: list, maxCapacity: int):
        self.items = items
        self.maxCapacity = maxCapacity

    def __len__(self):
        """
        :return: the total number of items in this knapsack problem.
        """
        return len(self.items)

    def getValue(self, zeroOneList: list) -> int:
        """
        Calculates the maximum value for fitness of the knapsack representation but only using items that do not
        exceed the maximum weight.
        """
        totalWeight = totalValue = 0

        for i in range(len(self.items)):
            if zeroOneList[i] == 0:
                continue
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                totalWeight += zeroOneList[i] * weight
                totalValue += zeroOneList[i] * value

        return totalValue

    def printItems(self, zeroOneList: list):
        """
        Prints the selected items in the list, while ignoring the items that will cause the weight to exceed the
        maximum.
        :param zeroOneList: a list of 0/1 values, 1 means that the item was selected.
        """

        totalWeight = totalValue = 0
        for i in range(len(self.items)):
            item, weight, value = self.items[i]
            if totalWeight + weight <= self.maxCapacity:
                if zeroOneList[i] > 0:
                    totalWeight += weight
                    totalValue += value
                    print(f'- Adding {item}: weight = {weight}, value = {value}, accumulated weight = {totalWeight}, '
                          f'accumulated value = {totalValue}')
        print(f'- Total weight = {totalWeight}, Total value = {totalValue}')


knapsack = Knapsack([
    ("map", 9, 150),
    ("compass", 13, 35),
    ("water", 153, 200),
    ("sandwich", 50, 160),
    ("glucose", 15, 60),
    ("tin", 68, 45),
    ("banana", 27, 60),
    ("apple", 39, 40),
    ("cheese", 23, 30),
    ("beer", 52, 10),
    ("suntan cream", 11, 70),
    ("camera", 32, 30),
    ("t-shirt", 24, 15),
    ("trousers", 48, 10),
    ("umbrella", 73, 40),
    ("waterproof trousers", 42, 70),
    ("waterproof overclothes", 43, 75),
    ("note-case", 22, 80),
    ("sunglasses", 7, 20),
    ("towel", 18, 12),
    ("socks", 4, 50),
    ("book", 30, 10)
], 400)


def knapsackFitness(individual) -> tuple:
    return knapsack.getValue(individual),


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
                 creator.Individual, toolbox.zeroOrOne, len(knapsack))

# Create random population operator
toolbox.register('populationCreator', tools.initRepeat,
                 list, toolbox.individualCreator)

toolbox.register('evaluate', knapsackFitness)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0 / len(knapsack))


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

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    print('Best Individual = ', hof.items[0])
    solution = hof.items[0]
    knapsack.printItems(solution)

    maxFitnessValues, meanFitnessValues = logbook.select('max', 'avg')
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()
