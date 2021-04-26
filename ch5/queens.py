import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class NQueensProblem:

    def __init__(self, numberOfQueens):
        """
        :param numberOfQueens: the number of queens in the problem.
        """
        self.numOfQueens = numberOfQueens

    def __len__(self):
        return self.numOfQueens

    def getViolationsCount(self, positions):
        """
        Calculates the number of violations in the given solution.
        Only diagonal violation need to be counted.
        :param positions: a list of indices corresponding to the positions of the queens in each row.
        :return: the number of violations
        """
        self.__checkPositionsLength(positions)

        violations = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                column1 = i
                row1 = positions[i]

                column2 = j
                row2 = positions[j]

                # queens are in a diagonal if the vertical and horizontal distances are equal
                if abs(column1 - column2) == abs(row1 - row2):
                    violations += 1
        return violations

    def plotBoard(self, positions):
        """
        Plots the positions of the queens on the board according to the given solution.
        :param positions: a list of indices corresponding to the positions of the queens in each row.
        """
        self.__checkPositionsLength(positions)

        fix, ax = plt.subplots()

        # start with the board's squares
        board = np.zeros((self.numOfQueens, self.numOfQueens))
        board[::2, 1::2] = 1
        board[1::2, ::2] = 1

        # draw the squares with two different colors
        ax.imshow(board, interpolation='none', cmap=mpl.colors.ListedColormap(['#ffc792', '#4c2f27']))

        # read the queen image thumbnail and give it a spread of 70% of the square dimensions
        queenThumbnail = plt.imread('queen-thumbnail.png')
        thumbnailSpread = 0.70 * np.array([-1, 1, -1, 1]) / 2  # spread is [left, right, bottom, top]

        for i, j in enumerate(positions):
            # place the thumbnail on the matching square
            ax.imshow(queenThumbnail, extent=[j, j, i, i] + thumbnailSpread)

        # show the row and column indexes
        ax.set(xticks=list(range(self.numOfQueens)), yticks=list(range(self.numOfQueens)))
        ax.axis('image')

        return plt

    def __checkPositionsLength(self, positions):
        if len(positions) != self.numOfQueens:
            raise ValueError('size of positions list should be equal to ', self.numOfQueens)


def main():
    nQueens = NQueensProblem(8)
    # a solution with 3 violations
    solution = [1, 2, 7, 5, 0, 3, 4, 6]
    print(f'Number of violations = {nQueens.getViolationsCount(solution)}')

    plot = nQueens.plotBoard(solution)
    plot.show()


if __name__ == '__main__':
    main()
