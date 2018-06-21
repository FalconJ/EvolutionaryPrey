import numpy as np
from enum import Enum


# func to generate individual
def generate_individual(
        get_fitness, upp_boundary, low_boundary, dimSize, grid, classi):
    genes, fitness = None, None
    if classi == 0:
        genes = np.random.uniform(low_boundary, upp_boundary, 2)
        fitness = get_fitness(genes)

    x, y, grid = create_coordinates(grid)

    if (x != -1 and y != -1):
        return Chromosome(genes, fitness, x, y, classi), grid
    else:
        return None, grid


# add individual to grid
def create_coordinates(grid):
    size = len(grid)
    for _ in range(10):
        x, y = np.random.randint(0, high=size), np.random.randint(0, high=size)
        if(grid[y][x] == 0):
            grid[y][x] = 1
            return x, y, grid
    return -1, -1, grid


# func generate population of prey
def generate_pop(pop_size, grid, dim, upp_boundary, low_boundary, get_fitness):
    population = []
    predator_size = int(np.floor(pop_size / 12))

    if predator_size == 0:
        predator_size += 1

    for _ in range(pop_size):
        individual, grid = generate_individual(
            get_fitness,
            upp_boundary,
            low_boundary,
            dim,
            grid,
            0)
        population.append(individual)

    for i in range(predator_size):
        predator_type = i % 2 + 1
        individual, grid = generate_individual(
            get_fitness,
            0,
            0,
            0,
            grid,
            predator_type)
        population.append(individual)

    return population, grid


# Genetic operators

def _mutate(genes, get_fitnesss, upp_boundary, low_boundary, pop_size):
    mutate_prob = 0.5
    pop_range = 1 / pop_size
    pop_chromosome = 1 / len(genes)

    if pop_range <= pop_chromosome:
        mutation_rate = np.random.uniform(pop_range, pop_chromosome)
    else:
        mutation_rate = np.random.uniform(pop_chromosome, pop_range)

    if(mutate_prob > mutation_rate):
        new_gene = np.random.uniform(low=low_boundary, high=upp_boundary)
        index = np.random.randint(0, high=len(genes))
        genes[index] = new_gene

    return genes, get_fitnesss(genes)


def _crossover(parent, otherParent, grid, get_fitness):
    childGenes = parent[:]

    if len(parent) <= 2 or len(otherParent) < 2:
        childGenes[1] = otherParent[1]
    else:
        length = np.randint(1, high=len(parent) - 2)
        start = np.randint(0, high=len(parent) - length)
        childGenes[start:start + length] = otherParent[start:start + length]

    fitness = get_fitness(childGenes)
    x, y, grid = create_coordinates(grid)

    if x != -1 and y != -1:
        # print("crossover")
        return Chromosome(childGenes, fitness, x, y, 0), grid
    else:
        return None, grid


# New offspring
def get_offspring(get_fitness, prey, neighbor, grid):
    return _crossover(prey.Genes, neighbor.Genes, grid, get_fitness)
    # return np.random.choice(usedStrategies)(prey, neighbor)


class Strategies(Enum):
    Create = 0,
    Mutate = 1,
    Crossover = 2


# Number of types of predators
class Classification(Enum):
    Prey = 0,
    Predator_1 = 1,
    Predator_2 = 2


class Chromosome:
    Genes = None
    Fitness = None
    x_coordinate = None
    y_coordinate = None
    classification = None


    def __init__(self, genes, fitness, x_coordinate, y_coordidnate, classification):
        self.Genes = genes
        self.Fitness = fitness
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordidnate
        self.classification = classification
