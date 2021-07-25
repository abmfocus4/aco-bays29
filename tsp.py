# Full Name: Meg Alapati
# Student #: 20747086
# Student ID: bmalapat

import math


class TSP:
    def __init__(self):
        super().__init__()

        self.bays29_coordinates = [
            (1150.0, 1760.0),  # 1
            (630.0, 1660.0),  # 2
            (40.0, 2090.0),  # 3
            (750.0, 1100.0),  # 4
            (750.0, 2030.0),  # 5
            (1030.0, 2070.0),  # 6
            (1650.0, 650.0),  # 7
            (1490.0, 1630.0),  # 8
            (790.0, 2260.0),  # 9
            (710.0, 1310.0),  # 10
            (840.0, 550.0),  # 11
            (1170.0, 2300.0),  # 12
            (970.0, 1340.0),  # 13
            (510.0, 700.0),  # 14
            (750.0, 900.0),  # 15
            (1280.0, 1200.0),  # 16
            (230.0, 590.0),  # 17
            (460.0, 860.0),  # 18
            (1040.0, 950.0),  # 19
            (590.0, 1390.0),  # 20
            (830.0, 1770.0),  # 21
            (490.0, 500.0),  # 22
            (1840.0, 1240.0),  # 23
            (1260.0, 1500.0),  # 24
            (1280.0, 790.0),  # 25
            (490.0, 2130.0),  # 26
            (1460.0, 1420.0),  # 27
            (1260.0, 1910.0),  # 28
            (360.0, 1980.0)  # 29
        ]

        self.num_cities = 29

    def calculate_weights(self):
        # initialize weights array
        weights = [[0 for x in range(self.num_cities)] for y in range(self.num_cities)]

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                weights[j][j] = 0

                # coordinates
                x1 = self.bays29_coordinates[i][0]
                y1 = self.bays29_coordinates[i][1]

                x2 = self.bays29_coordinates[j][0]
                y2 = self.bays29_coordinates[j][1]

                # calculate euclidean distances
                weights[i][j] = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))

                # weights from city i to j is the same as the weight of city j to i
                weights[j][i] = weights[i][j]
        return weights
