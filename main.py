# Full Name: Meg Alapati
# Student #: 20747086
# Student ID: bmalapat

import tsp
import random
import math
from matplotlib import pyplot as plt
import numpy as np
from statistics import mean


# todo: report
# todo: graphs
# todo: max_iterations change

def get_best_tour(ants, weights):
    best = calculate_cost(ants[0], weights)
    best_ant = 0

    # find the ant with the best tour so far & cost of tour
    for k in range(1, len(ants)):
        current = calculate_cost(ants[k], weights)
        if current < best:
            best = current
            best_ant = k

    return ants[best_ant]


def calculate_cost(tour, weights):
    cost = 0
    for i in range(len(tour) - 1):
        cost += weights[tour[i]][tour[i + 1]]
    return cost


def online_update(ants, pheromones, weights):
    try:
        for i in range(len(pheromones)):
            for j in range(i + 1, len(pheromones[i])):
                for k in range(len(ants)):
                    # decrease pheromone decay constant
                    pheromones[i][j] *= (1 - rho)
                    # add new pheromone concentration if path is visited
                    if visited_path(i, j, ants[k]):
                        pheromones[i][j] += q / calculate_cost(ants[k], weights)
                    # # pheromone is always equal or greater than decay constant
                    # if pheromones[i][j] < rho:
                    #     pheromones[i][j] = rho

                # adjacency preserve
                pheromones[j][i] = pheromones[i][j]
    except:
        return


def visited_path(node1, node2, edges):
    try:
        node1_idx = edges.index(node1)
        node2_idx = edges.index(node2)
    except:
        node1_idx = 0
        node2_idx = 0

        last_node = len(edges) - 1
    # check if edge connected two nodes is visited
    if (node1_idx - 1 == node2_idx) or (node1_idx + 1 == node2_idx) or (node1_idx == 0 and node2_idx == last_node) \
            or (node2_idx == 0 and node1_idx == last_node):
        return True
    else:
        return False


def offline_update(ants, pheromones, weights):
    for i in range(len(pheromones)):
        for j in range(len(pheromones[i])):
            for k in range(population):
                pheromones[i][j] = (pheromones[i][j] * (1 - rho)) + (q / calculate_cost(ants[k], weights)) * rho
                # preserve adjacency
                pheromones[j][i] = pheromones[i][j]


# return a random tour with 29 cities
def get_tour(start_city, pheromones, weights):
    tour = [0 for _ in range(num_cities)]
    visited = [False for k in range(num_cities)]

    tour[0] = start_city
    if start_city < num_cities:
        visited[start_city] = True

    for i in range(last_city):
        next_city = tour[i]
        _next = select_next(next_city, visited, pheromones, weights)
        tour[i + 1] = _next
        visited[_next] = True

    return tour


def select_next(city, visited, pheromones, weights):
    probability = transition_prob(city, pheromones, weights, visited)
    cum_prob = [0 for _ in range(len(probability) + 1)]

    # calculate cumulative probability from of each city
    for i in range(len(probability)):
        cum_prob[i + 1] = cum_prob[i] + probability[i]

    # finding the next city to visit within the range of probability
    random_fraction = random.random()
    for j in range(len(cum_prob) - 1):
        if cum_prob[j] <= random_fraction < cum_prob[j + 1]:
            return j


def transition_prob(city, pheromones, weights, visited):
    probability = [0 for _ in range(num_cities)]
    total = 0

    for i in range(len(probability)):
        # already visited - don't want to visit again
        if i == city or visited[i]:
            probability[i] = 0
        # apply transition rule
        else:
            if city < len(pheromones) and i < len(pheromones[city]):
                probability[i] = math.pow(pheromones[city][i], alpha) * math.pow((1 / weights[i][city]), beta)

        # calculate total probability
        total += probability[i]

    balanced_prob = [0 for _ in range(num_cities)]
    # update probabilities with total
    for i in range(num_cities):
        if total != 0:
            balanced_prob[i] = probability[i] / total

    return balanced_prob


def print_output(tour, cost, iteration):
    print('Best Tour Found by ACO for Bays29:')
    best_tour = [x + 1 for x in tour]
    print(best_tour)
    print('Cost of Best Tour is ' + str(cost) + ' found at Iteration ' + str(iteration) + ' by an Ant')
    #
    # # creating Map of tour
    # coordinates = [cities.bays29_coordinates[tour[i]] for i in range(len(tour))]
    #
    # x = np.array([x[0] for x in coordinates])
    # y = np.array([y[1] for y in coordinates])
    #
    # plt.figure()
    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='b')
    #
    # plt.scatter(x, y, color='r')
    # for i in range(len(x)):
    #     label = ''
    #     if i == 0:
    #         label = 'START'
    #     elif i == len(x) - 1:
    #         label = 'END'
    #     plt.annotate(label, (x[i], y[i]), ha='left', textcoords="offset points", xytext=(-15, -15))
    #
    # plt.xlabel('X-Coordinate')
    # plt.ylabel('Y-Coordinate')
    # plt.title('Best Tour')
    #
    # plt.savefig('best_tour')
    #
    # print('')
    # print('Tour Map saved as best_tour.png')


if __name__ == '__main__':
    # bays29 parameters
    cities = tsp.TSP()
    weights = cities.calculate_weights()
    num_cities = cities.num_cities
    last_city = num_cities - 1

    # aco parameters
    print('Ant Colony Optimization - TSP Bays29')
    print('')

    # y axis
    # pop = [i for i in range(1, 30, 2)]
    # pop = [i for i in range(1, 4)]
    # rho_list = [0.001, 0.05, 0.01, 0.5, 0.1]
    rho_list = [0.001, 0.01, 0.999]

    # plot 1
    cost_test1 = []
    time1 = []
    # plot 2
    cost_test2 = []
    time2 = []
    # plot 3
    cost_test3 = []
    time3 = []


    # trails
    # num_trails = 10

    # default = int(input('Enter 1 to run ACO with default parameters and 0 to input custom parameters'))
    # print('')
    # for p in range(len(pop)):
    for r in range(len(rho_list)):
        # pop_cost = []
        # rho_cost = []
        # for i in range(num_trails + 1):
        # population = pop[p]
        population = 15
        # rho = 0.5
        rho = rho_list[r]
        alpha = 1
        beta = 5
        q = 2000
        max_iterations = 2000
        online = True
        # max_iterations = 100

        # if default:
        #     population = 15
        #     rho = 0.5
        #     alpha = 1
        #     beta = 5
        #     q = 2000
        #     max_iterations = 1000
        #     online = True
        #     # max_iterations = 100
        # # population - pick 2 good ones for range - done
        # # rho - pick 3 good ones - 0.9, 0.001, (0.01, 0.99)
        # # Q pick 3 values
        # # beta pick 3 values
        # # online turn off
        # else:
        #     population = int(input('Enter population of ant colony:'))
        # if population > 1000 or population <= 0:
        #     population = 1000
        #     rho = float(input('Enter pheromone decay constant value:'))
        # if rho <= 0 or rho >= 1:
        #     rho = 0.5
        #     alpha = float(input('Enter value of alpha: '))
        # if alpha > 10 or alpha <= 0:
        #     alpha = 1
        #     beta = float(input('Enter value of beta: '))
        # if beta > 10 or beta <= 0:
        #     beta = 5
        #     q = float(input('Enter value of Q: '))
        # if q <= 0:
        #     q = 2000
        #     online = bool(input('Enter 1 for enabling and 0 for disabling online pheromone update'))
        #     max_iterations = int(input('Enter the number of iterations ACO needs to run for: '))
        # if max_iterations <= 0:
        #     max_iterations = 100
        #     print('')

        print('Running ACO with the following parameters: ')
        print('')

        print('population: ' + str(population))
        print('pheromone decay constant: ' + str(rho))
        print('online pheromone update: ' + str(online))
        print('alpha: ' + str(alpha))
        print('beta: ' + str(beta))
        print('Q: ' + str(q))
        print('max iterations: ' + str(max_iterations))
        print('')

        print('Ant Colony Construction in Progress...')
        print('')

        ants = [0 for _ in range(population)]
        pheromones = [[0 for _ in range(num_cities)] for y in range(num_cities)]

        # initialize ant k's tour with random cities
        for k in range(population):
            ants[k] = []
            for i in range(num_cities):
                # get random city not in tour
                city = random.randint(0, last_city)
                while city in ants[k]:
                    city = random.randint(0, last_city)
                # add city to tour
                ants[k].append(city)

        # set pheromone values to default
        for i in range(num_cities):
            for j in range(num_cities):
                pheromones[i][j] = rho

        best_tour = get_best_tour(ants, weights)
        best_cost = calculate_cost(best_tour, weights)

        print('Initial Tour Cost: ' + str(best_cost))

        iterations = 0
        best_iteration = 0

        while iterations < max_iterations:
            if iterations % 100 == 0 and iterations:
                print("Iterations Completed: " + str(iterations))

            if online:
                online_update(ants, pheromones, weights)
            offline_update(ants, pheromones, weights)

            for i in range(len(ants)):
                ants[i] = get_tour(ants[i][0], pheromones, weights)

            new_best_tour = get_best_tour(ants, weights)
            new_best_cost = calculate_cost(new_best_tour, weights)

            if new_best_cost < best_cost:
                best_cost = new_best_cost
                best_tour = new_best_tour
                # print('Cost of Best Tour so far: ' + str(best_cost))
                best_iteration = iterations
                if r == 0:
                    time1.append(iterations)
                    cost_test1.append(best_cost)
                if r == 1:
                    time2.append(iterations)
                    cost_test2.append(best_cost)
                if r == 2:
                    time3.append(iterations)
                    cost_test3.append(best_cost)


            # if iterations % 10 == 0 and iterations:
            #     if r == 0:
            #         time1.append(iterations)
            #         cost_test1.append(best_cost)
            #     # if r == 1:
            #     #     time2.append(iterations)
            #     #     cost_test2.append(new_best_cost)
            #     if r == 1:
            #         time3.append(iterations)
            #         cost_test3.append(best_cost)

            iterations += 1

        print('')
        # pop_cost.append(best_cost)
        # rho_cost.append(best_cost)
        print_output(best_tour, best_cost, best_iteration)

        # cost.append(mean(pop_cost))
        # cost.append(mean(rho_cost))

    # x-axis
    # print(pop)
    # print(time)

    # y-axis
    # print(cost)
    # print(cost_test)

    plt.figure()
    # plt.plot(pop, cost, color='b')
    # plt.scatter(pop, cost, color='r')

    plt.plot(time1, cost_test1)
    plt.plot(time2, cost_test2)
    plt.plot(time3, cost_test3)

    # plt.scatter(time, cost_test)

    # for i in range(len(cost)):
    #     plt.annotate((rho_list[i], cost[i]), (rho_list[i], rho_list[i]), ha='center', textcoords="offset points",
    #                  xytext=(5, 5))

    plt.xlabel('Iterations')
    plt.ylabel('Tour Cost')
    plt.title('change in performance with pheromone decay constant')
    plt.legend(["rho = 0.001, rho = 0.01", "rho = 0.9"])

    plt.savefig('rho_vs_cost-step')
