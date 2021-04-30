import numpy as np


class NurseSchedulingProblem:
    """
    This class represents the nurse scheduling problem.
    """

    def __init__(self, hardConstraintPenalty):
        """
        :param hardConstraintPenalty: the penalty for violating a hard-constraint.
        """
        self.hardConstraintPenalty = hardConstraintPenalty

        # list of nurses
        self.nurses = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')

        # nurses shift preferences: morning, evening, night
        self.shiftPreference = (
            (1, 0, 0),
            (1, 1, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (1, 1, 1)
        )

        # min and max number of nurses allowed for each shift - morning, evening, night
        self.shiftMin = (2, 2, 1)
        self.shiftMax = (3, 4, 2)

        # max shifts per week allowed for each nurse
        self.maxShiftsPerWeek = 5

        # number of weeks we create the schedule
        self.weeks = 1

        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = 7 * self.shiftPerDay

    def __len__(self):
        """
        :return: the number of shifts in the schedule
        """
        return len(self.nurses) * self.shiftsPerWeek * self.weeks

    def getCost(self, schedule) -> int:
        """
        Calculates the total cost of the various violations in the given schedule.
        :param schedule: a list a binary values describing the given schedule.
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError('size of schedule list should be equal to ', self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each
        nurseShiftsDict = self.getNurseShifts(schedule)

        # count the various violations
        consecutiveShiftViolations = self.countConsecutiveShiftViolations(nurseShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(nurseShiftsDict)[1]
        nursesPerShiftViolations = self.countNursesPerShiftViolations(nurseShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)

        # calculate the cost of violations
        hardConstraintViolations = consecutiveShiftViolations + nursesPerShiftViolations + shiftsPerWeekViolations
        softConstraintViolations = shiftPreferenceViolations

        return self.hardConstraintPenalty * hardConstraintViolations + softConstraintViolations

    def getNurseShifts(self, schedule: list) -> dict:
        """
        Converts the entire schedule into a dictionary with a separate schedule for each nurse.
        :param schedule: a list of binary values describing the given schedule.
        :return: a dictionary with each nurse as a key and the corresponding shifts as the value.
        """
        shiftsPerNurse = self.__len__() // len(self.nurses)
        nurseShiftsDict = {}
        shiftIndex = 0

        for nurse in self.nurses:
            nurseShiftsDict[nurse] = schedule[shiftIndex:shiftIndex + shiftsPerNurse]
            shiftIndex += shiftsPerNurse

        return nurseShiftsDict

    def countConsecutiveShiftViolations(self, nurseShiftsDict: dict) -> int:
        """
        Counts consecutive shift violations in the schedule.
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse.
        :return: count of violations found.
        """
        violations = 0
        # iterate over the shifts of each nurse
        for nurseShifts in nurseShiftsDict.values():
            # look for two consecutive 1s
            for shift1, shift2 in zip(nurseShifts, nurseShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def countShiftsPerWeekViolations(self, nurseShiftsDict: dict) -> tuple:
        """
        Counts the max-shifts-per-week violations in the schedule.
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse.
        :return: count of violations found.
        """
        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each nurse
        for nurseShifts in nurseShiftsDict.values():
            # iterate over the shifts of each weeks
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # count all 1s over the week
                weeklyShifts = sum(nurseShifts[i:i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek

        return weeklyShiftsList, violations

    def countNursesPerShiftViolations(self, nurseShiftsDict: dict) -> tuple:
        """
        Counts the number-of-nurses-per-shift violations in the schedule.
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse.
        :return: count of violations found.
        """

        # sum the shifts over all nurses
        totalPerShiftList = [sum(shift) for shift in zip(*nurseShiftsDict.values())]

        violations = 0
        # iterator over all shifts and count violations
        for shiftIndex, numOfNurses in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay
            if numOfNurses > self.shiftMax[dailyShiftIndex]:
                violations += numOfNurses - self.shiftMax[dailyShiftIndex]
            elif numOfNurses < self.shiftMin[dailyShiftIndex]:
                violations += self.shiftMin[dailyShiftIndex] - numOfNurses

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, nurseShiftsDict: dict) -> int:
        """
        Counts the nurse-preferences violations in the schedule.
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse.
        :return: count of violations found.
        """
        violations = 0
        for nurseIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            # iterate over the shifts and compare preferences
            shifts = nurseShiftsDict[self.nurses[nurseIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations

    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details.
        :param schedule: a list of binary values describing the given schedule.
        """
        nurseShiftsDict = self.getNurseShifts(schedule)

        print('- Schedule for each nurse:')
        for nurse in nurseShiftsDict:  # all shifts of a single nurse
            print('- ', nurse, ':', nurseShiftsDict[nurse])

        print('- Consecutive shift violations = ', self.countConsecutiveShiftViolations(nurseShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(nurseShiftsDict)
        print('- Weekly shifts = ', weeklyShiftsList)
        print('- Shifts per week violations = ', violations)
        print()

        totalPerShiftList, violations = self.countNursesPerShiftViolations(nurseShiftsDict)
        print('- Nurses per shift = ', totalPerShiftList)
        print('- Nurses per shift violations = ', violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)
        print('- Shift preference violations = ', shiftPreferenceViolations)
        print()
