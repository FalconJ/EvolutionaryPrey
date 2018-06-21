import unittest
import genetic
import numpy as np
import math
import matplotlib.pyplot as plt

# math functions
sin = np.sin
cos = np.cos
sqrt = np.sqrt
pi = np.pi
fabs = np.fabs
pows = np.power
exp = np.exp
euler = math.e


# Fitness equations
def first_func(genes):
    return -1 * pows(genes[0], 2) + pows(genes[1], 2)


def second_func(genes):
    x = genes[0]
    y = genes[1]

    x += 2
    fitNumber = pows(x, 2) + pows(y, 2)
    return -1 * fitNumber


def get_fitness(genes):
    first_fit = first_func(genes)
    second_fit = second_func(genes)

    return [first_fit, second_fit]


# movement
def move(animal, grid):
    # possible directions
    N, NE, E, SE, S, SW, W, NW = 0, 1, 2, 3, 4, 5, 6, 7
    animal_x = animal.x_coordinate
    animal_y = animal.y_coordinate
    size = len(grid)

    for i in range(10):
        direction = int(np.random.randint(0, high=8))
        new_x, new_y = animal_x, animal_y

        if direction == N:
            new_y = (animal_y - 1) % size
        elif direction == NE:
            new_x = (animal_x + 1) % size
            new_y = (animal_y - 1) % size
        elif direction == E:
            new_x = (animal_y + 1) % size
        elif direction == SE:
            new_x = (animal_x + 1) % size
            new_y = (animal_y + 1) % size
        elif direction == S:
            new_y = (animal_y + 1) % size
        elif direction == SW:
            new_x = (animal_x - 1) % size
            new_y = (animal_y + 1) % size
        elif direction == W:
            new_x = (animal_x - 1) % size
        elif direction == NW:
            new_x = (animal_x - 1) % size
            new_y = (animal_y - 1) % size

        if not detect_collision(new_x, new_y, grid):
            grid[animal_y][animal_x] = 0
            grid[new_y][new_x] = 1
            animal.x_coordinate = new_x
            animal.y_coordinate = new_y
            break
    return animal, grid


def detect_collision(x, y, grid):
    return grid[y][x] == 1


def get_neighbors(individual, population):
    # result = {}
    neighbors = []
    pop_coordinates = {}

    for animal in population:
        pop_coordinates[(animal.x_coordinate, animal.y_coordinate)] = animal

    for x, y in [
        (individual.x_coordinate + i, individual.y_coordinate + j)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0]:
                if(x, y) in pop_coordinates:
                    # result[x, y] = pop_coordinates[(x, y)]
                    neighbors.append(pop_coordinates[(x, y)])
    return neighbors


def mutate(population, get_fitness, upp_boundary, low_boundary):
    for prey in population:
        prey.Genes, prey.Fitness = genetic._mutate(
            prey.Genes,
            get_fitness,
            upp_boundary,
            low_boundary,
            len(population))
    return population


def breeding(population, grid, get_fitness):
    new_population = []
    for prey in population:
        hood = get_neighbors(prey, population)

        if hood:
            for neighbor in hood:
                    offspring, grid = genetic.get_offspring(
                        get_fitness,
                        prey,
                        neighbor,
                        grid)
                    if offspring:
                        new_population.append(offspring)
    population += new_population
    return population, grid


def prey_moving(prey_pop, grid):
    for prey in prey_pop:
        prey, grid = move(prey, grid)
    return prey_pop, grid


def hunting(pred_pop, prey_pop, grid):
    for wolf in pred_pop:
        targets = get_neighbors(wolf, prey_pop)
        if targets:
            # print([x.Fitness for x in targets])
            if wolf.classification == 1:
                prey_selected = min(targets, key=lambda x: x.Fitness[0])
            else:
                prey_selected = min(targets, key=lambda x: x.Fitness[1])
            prey_pop.remove(prey_selected)
            grid[wolf.y_coordinate][wolf.x_coordinate] = 0
            wolf.y_coordinate = prey_selected.y_coordinate
            wolf.x_coordinate = prey_selected.x_coordinate
        else:
            wolf, grid = move(wolf, grid)
    return pred_pop, prey_pop, grid


def plot_pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    '''Pareto frontier selection process'''
    sorted_list = sorted(
        [[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    plt.scatter(Xs, Ys)
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y)
    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.show()


class PredatorVsPrey(unittest.TestCase):
    def test(self):
        size = 30
        dim = 2
        pop_size = 250
        upp_boundary = 50
        low_boundary = -50
        num_prey_prefer = 100
        num_generation = 200

        grid = np.zeros((size, size), dtype=np.int)

        population, grid =\
            genetic.generate_pop(
                pop_size, grid, dim, upp_boundary, low_boundary, get_fitness)

        for generation in range(num_generation):
            prey = []
            predator = []

            for x in population:
                if x.classification == 0:
                    prey.append(x)
                else:
                    predator.append(x)
            prey, grid = prey_moving(prey, grid)
            prey = mutate(prey, get_fitness, upp_boundary, low_boundary)
            prey, grid = breeding(prey, grid, get_fitness)

            num_iteration = (len(prey) - num_prey_prefer) / len(predator)
            iterations = int(np.ceil(num_iteration)) + 4
            print("Before hunt: ", len(prey))
            for i in range(iterations):
                predator, prey, grid = hunting(predator, prey, grid)

            print("After hunt: ", len(prey))
            population = prey + predator
            print(len(prey)," + ", len(predator)," = ", len(population))

        X, Y = [], []
        for x in population:
            if x.classification == 0:

                print("Genes:", x.Genes)
                X.append(x.Fitness[0])
                Y.append(x.Fitness[1])
        plot_pareto_frontier(X, Y)


if __name__ == "__main__":
    unittest.main()
