import unittest
import numpy as np
from nurses import NurseSchedulingProblem


class NursesTestSuite(unittest.TestCase):
    def test_length(self):
        nsp = NurseSchedulingProblem(10)
        self.assertEqual(168, len(nsp))

    def test_violations(self):
        nsp = NurseSchedulingProblem(10)
        randomSolution = np.random.randint(2, size=len(nsp))
        self.assertTrue(nsp.getCost(randomSolution) > 0)
