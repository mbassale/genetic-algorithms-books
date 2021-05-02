import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GraphColoringProblem:
    """
    This class encapsulates the Graph Coloring Problem
    """

    def __init__(self, graph: nx.Graph, hardConstraintPenalty: int):
        """
        :param graph: a NetworkX graph to be colored.
        :param hardConstraintPenalty: penalty for coloring violation.
        """
        # initialize instance variables
        self.graph = graph
        self.hardConstraintPenalty = hardConstraintPenalty

        # a list of nodes in the graph
        self.nodeList = list(self.graph.nodes)

        # adjacency matrix of the nodes
        # matrix[i, j] = 1 if nodes are connected, 0 otherwise
        self.adjMatrix = nx.adjacency_matrix(graph).todense()

    def __len__(self) -> int:
        """
        :return: the number of nodes in the graph
        """
        return nx.number_of_nodes(self.graph)

    def getCost(self, colorArrangement) -> int:
        """
        Calculates the cost of the suggested color arrangement.
        :param colorArrangement: a list of integers representing the suggested color arrangement.
        :return: Calculated cost of the arrangement.
        """
        return self.hardConstraintPenalty * self.getViolationsCount(colorArrangement) + self.getNumberOfColors(
            colorArrangement)

    def getViolationsCount(self, colorArrangement) -> int:
        """
        Calculates the number of violations in the given color arrangement.
        :param colorArrangement: a list of integers representing the color arrangement.
        :return: the number of violations
        """
        self.checkColorArrangement(colorArrangement)

        violations = 0
        # iterate over every pair of nodes and find if they are adjacent and share the same color
        for i in range(len(colorArrangement)):
            for j in range(i + 1, len(colorArrangement)):
                if self.adjMatrix[i, j]:
                    if colorArrangement[i] == colorArrangement[j]:
                        violations += 1
        return violations

    def getNumberOfColors(self, colorArrangement) -> int:
        """
        returns the number of different colors in the suggested color arrangement.
        :param colorArrangement: a list of integers representing the color arrangement.
        :return: number of different colors
        """
        return len(set(colorArrangement))

    def plotGraph(self, colorArrangement):
        """
        Plots the graph with the nodes colored according to the given color arrangement.
        :param colorArrangement: a list of integers representing the color arrangement.
        """
        self.checkColorArrangement(colorArrangement)

        # create a list of the unique colors in the arrangement
        colorList = list(set(colorArrangement))

        # create the actual colors for the integers in the color list
        colors = plt.cm.rainbow(np.linspace(0, 1, len(colorList)))

        # iterate over the nodes, and give each one of them its corresponding color
        colorMap = []
        for i in range(self.__len__()):
            color = colors[colorList.index(colorArrangement[i])]
            colorMap.append(color)

        # plot the nodes with their labels and matching colors
        nx.draw_kamada_kawai(self.graph, node_color=colorMap, with_labels=True)
        return plt

    def checkColorArrangement(self, colorArrangement: list):
        if len(colorArrangement) != self.__len__():
            raise ValueError('size of color arrangement should be equal to ', self.__len__())


def main():
    # create a problem instance with petersen graph
    gcp = GraphColoringProblem(nx.petersen_graph(), 10)

    # generate a random solution with up to 5 different colors
    solution = np.random.randint(5, size=len(gcp))

    print('- solution = ', solution)
    print('- number of colors = ', gcp.getNumberOfColors(solution))
    print('- number of violations = ', gcp.getViolationsCount(solution))
    print('- cost = ', gcp.getCost(solution))

    plot = gcp.plotGraph(solution)
    plot.show()


if __name__ == '__main__':
    main()
