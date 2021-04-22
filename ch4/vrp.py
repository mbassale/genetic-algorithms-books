import random
import numpy as np
import matplotlib.pyplot as plt
from tsp import TSP


class VRP:

    def __init__(self, tspName, numOfVehicles, depotIndex):
        """
        Creates an instance of a VRP.
        :param tspName: name of the underlying TSP.
        :param numOfVehicles: number of vehicles used.
        :param depotIndex: the index of the TSP city used as depot location.
        """
        self.tsp = TSP(tspName)
        self.numOfVehicles = numOfVehicles
        self.depotIndex = depotIndex

    def __len__(self):
        """
        returns the number of indices used to internally represent the VRP.
        """
        return len(self.tsp) + self.numOfVehicles - 1

    def getRoutes(self, indices):
        """
        breaks the list of given indices into separate routes by detecting 'separator' indices.
        :param indices: list of indices, including 'separator' indices.
        :return: a list of routes, each route being a list of location indices from the TSP problem.
        """

        routes = []
        route = []

        for i in indices:

            # skip depot index
            if i == self.depotIndex:
                continue

            # index is part of the current route
            if not self.isSeparatorIndex(i):
                route.append(i)

            # separator index, the route is complete
            else:
                routes.append(route)
                route = []

        # append the last route
        if route or self.isSeparatorIndex(i):
            routes.append(route)

        return routes

    def isSeparatorIndex(self, index):
        """
        Finds if current index is a separator index.
        :param index: location index.
        :return: True if the given index is a separator.
        """
        # check if the index is larger than the number of locations
        return index >= len(self) - (self.numOfVehicles - 1)

    def getRouteDistance(self, indices):
        """
        Calculates the total distance of the path that starts at the depot location and goes through the cities.
        :param indices: a list of ordered city indices describing the given path.
        :return: total distance of the path described by the given indices.
        """
        if not indices:
            return 0

        # find the distance between the depot location and the first city
        distance = self.tsp.distances[self.depotIndex][indices[0]]

        # find the distance between the depot location and the last city
        distance += self.tsp.distances[indices[-1]][self.depotIndex]

        # add the distances of between the cities along the route
        for i in range(len(indices) - 1):
            distance += self.tsp.distances[indices[i]][indices[i + 1]]

        return distance

    def getTotalDistance(self, indices):
        """
        Calculates the combined distance of the various paths described by the given indices.
        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: combined distance of various paths described by the given indices.
        """
        totalDistance = 0
        for route in self.getRoutes(indices):
            routeDistance = self.getRouteDistance(route)
            # print(f'- Route distance = {routeDistance}')
            totalDistance += routeDistance
        return totalDistance

    def getMinDistance(self, indices):
        """
        Calculates the shortest distance of the various paths described by the given indices.
        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: min distance of the various paths described by the given indices.
        """
        minDistance = 0
        for route in self.getRoutes(indices):
            routeDistance = self.getRouteDistance(route)
            minDistance = min(routeDistance, minDistance)
        return minDistance

    def getMaxDistance(self, indices):
        """
        Calculates the max distance among the distances of the various paths described by the given indices.
        :param indices: a list of ordered city indices and separator indices describing one of more paths.
        :return: max distance of the various paths described by the given indices.
        """
        maxDistance = 0
        for route in self.getRoutes(indices):
            routeDistance = self.getRouteDistance(route)
            # print(f'- Route distance = {routeDistance}')
            maxDistance = max(routeDistance, maxDistance)
        return maxDistance

    def getAvgDistance(self, indices):
        """
        Calculates the average distance among the distance of the various paths described by the given indices.
        Does not consider empty paths.
        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: average distance among the distances of the various paths described by the indices.
        """
        routes = self.getRoutes(indices)
        totalDistance = 0
        numberOfPaths = 0
        for route in routes:
            if route:  # consider non-empty routes
                routeDistance = self.getRouteDistance(route)
                # print(f'- Route distance = {routeDistance}')
                totalDistance += routeDistance
                numberOfPaths += 1
        return totalDistance / numberOfPaths

    def plotData(self, indices):
        """
        Breaks the list of indices into separate routes and plot each route in a different color.
        :param indices: a list of ordered city indices and separator indices describing one or more paths.
        :return: a plot object.
        """

        # plot the cities
        plt.scatter(*zip(*self.tsp.locations), marker='.', color='red')

        # mark the depot location with a large 'X'
        depot = self.tsp.locations[self.depotIndex]
        plt.plot(depot[0], depot[1], marker='x', markersize=10, color='green')

        # break the indices to separate routes and plot each route in a different color
        routes = self.getRoutes(indices)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.numOfVehicles)))
        for route in routes:
            route = [self.depotIndex] + route + [self.depotIndex]
            stops = [self.tsp.locations[i] for i in route]
            plt.plot(*zip(*stops), linestyle='-', color=next(color))

        return plt


def main():
    vrp = VRP('bayg29', 3, 12)

    # generate random solution and evaluation
    randomSolution = random.sample(range(len(vrp)), len(vrp))
    print('Random solution = ', randomSolution)
    print('Route breakdown =', vrp.getRoutes(randomSolution))
    print('Max distance = ', vrp.getMaxDistance(randomSolution))

    plot = vrp.plotData(randomSolution)
    plot.show()


if __name__ == '__main__':
    main()
