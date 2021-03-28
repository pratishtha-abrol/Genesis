import new_client as server
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import json, os

SECRET_KEY='sjLXXU2W66Pc31A1FUe4Fz2mc0k15MTlinvDy1l0Q3GlE5sIEX'
FILE_NAME = 'output.txt'

# provided weights vector
test_file = open('overfit.txt','r')
overfit= test_file.readline().strip('[]').split(',')
overfit_vector=[]
for i in overfit:
    overfit_vector.append(float(i))


def write_file(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


POPULATION_SIZE = 30
CHROMOSOME_SIZE = len(overfit_vector)
GENERATIONS = 10
MATING_POOL_SIZE = 10
TRAIN_FACTOR = 0.7

# UNIVERSAL_DICT = {
#     'generation': [],
#     'weights': [],
#     'train_error': [],
#     'validation_error': [],
#     'fitness': []
# }
UNIVERSAL_DICT = []

DATA_LIST = []

CHILD_DICT = []
MATING_POOL = []

def initial_population():
    first_population = np.zeros((POPULATION_SIZE, CHROMOSOME_SIZE))
    
    for i in range(POPULATION_SIZE):
        for index in range(CHROMOSOME_SIZE):
            vary = 0
            mutation_prob = random.randint(0, 10)
            if mutation_prob < 3:
                if index <= 4:
                    vary = 1 + random.uniform(-0.05, 0.05)
                else:
                    vary = random.uniform(0, 1)
                rem = overfit_vector[index]*vary

                if abs(rem) < 10:
                    first_population[i][index] = rem
                elif abs(first_population[i][index]) >= 10:
                    first_population[i][index] = random.uniform(-1,1)

    return first_population

    # PREVIOUS METHODS
    '''
    parent_population = np.zeros((POPULATION_SIZE, CHROMOSOME_SIZE))
    for i in range(POPULATION_SIZE):
        temp = np.copy(overfit_vector)
        parent_population[i] = np.copy(mutate(temp, 0.4, 0.1))

    print(parent_population)            
    return parent_population
    '''

def mutate(temp, prob, mutate_range):
    vector = np.copy(temp)
    for i in range(len(vector)):
        fact=random.uniform(-mutate_range, mutate_range)
        vector[i] = np.random.choice([vector[i]*(fact+1), vector[i]], p=[prob,1-prob])
        if(vector[i]<-10) :
            vector[i]=-10
        elif(vector[i]>10) :
            vector[i]=10
            
    return vector


def calculate_fitness(population, generation):
    temp = []
    fitness_arr = np.empty((POPULATION_SIZE, 3))

    for i in range(POPULATION_SIZE):
        error = server.get_errors(SECRET_KEY, list(population[i]))
        # error = [np.random.randint(0, 5),np.random.randint(0,5)]
        fitness = abs(error[0]*TRAIN_FACTOR + error[1]) 
        fitness_arr[i][0] = error[0]
        fitness_arr[i][1] = error[1]
        fitness_arr[i][2] = fitness
        row = {
            # 'generation': generation,
            'weights': list(population[i]),
            'train_error': error[0],
            'validation_error': error[1],
            'fitness': fitness
        }
        data_row = {
            'generation': generation,
            'weights': list(population[i]),
            'train_error': error[0],
            'validation_error': error[1],
            'fitness': fitness
        }
        DATA_LIST.append(data_row)
        temp.append(row)
        # UNIVERSAL_DICT["generation"].append(0)
        # UNIVERSAL_DICT["weights"].append(list(population[i]))
        # UNIVERSAL_DICT["train_error"].append(error[0])
        # UNIVERSAL_DICT["validation_error"].append(error[1])
        # UNIVERSAL_DICT["fitness"].append(fitness)
    UNIVERSAL_DICT.append(temp)
    pop_fit = np.column_stack((population, fitness_arr))
    pop_fit = pop_fit[np.argsort(pop_fit[:,-1])]
    return pop_fit


def create_mating_pool(population_fitness, generation):
    # sort = sorted(UNIVERSAL_DICT[generation], key = lambda i: i['fitness'], reverse=True)
    # print(sort)
    # mating_pool = sort[:10]
    # return mating_pool
    # population_fitness = population_fitness[np.argsort(population_fitness[:,-1])]
    population_fitness = sorted(population_fitness, key=lambda x:x[-1])
    mating_pool = population_fitness[:MATING_POOL_SIZE]
    MATING_POOL.append(mating_pool)
    return mating_pool


def breed(mating_pool, generation):
    mating_pool = np.array(mating_pool)
    mating_pool = mating_pool[:, :-3]
    children = []
    children_dict = []
    for i in range( int(POPULATION_SIZE/2)):
        # parent1 = mating_pool[generation]["weights"][random.randint(0, MATING_POOL_SIZE-1)]
        # parent2 = mating_pool[generation]["weights"][random.randint(0, MATING_POOL_SIZE-1)]
        parent1 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        parent2 = mating_pool[random.randint(0, MATING_POOL_SIZE-1)]
        # print(parent1, parent2)
        child1, child2 = sb_crossover(parent1, parent2)
        child1 = mutation(child1)
        child2 = mutation(child2)
        children.append(child1)
        children.append(child2)
        dict_element1 = {
            'child': list(child1),
            'parent1': list(parent1),
            'parent2': list(parent2)
        }
        dict_element2 = {
            'child': list(child2),
            'parent1': list(parent1),
            'parent2': list(parent2)
        }
        children_dict.append(dict_element1)
        children_dict.append(dict_element2)

    CHILD_DICT.append(children_dict)
    return children 


def sb_crossover(parent1, parent2):
    child1 = np.empty(11)
    child2 = np.empty(11)
    u = random.random() 
    n_c = 3 
    flag = 0
    if (u < 0.5):
        beta = (2 * u)**((n_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((n_c + 1)**-1)
    p = random.random()
    if p < 0.3:
        flag = 1
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)
    child_1 = np.copy(child1)
    child_2 = np.copy(child2)
    if flag == 1:
        thresh = np.random.randint(CHROMOSOME_SIZE)
        child_1[thresh:CHROMOSOME_SIZE] =  child2[thresh:CHROMOSOME_SIZE]
        child_2[thresh:CHROMOSOME_SIZE] =  child1[thresh:CHROMOSOME_SIZE]
    return child_1, child_2

def sp_crossover(parent1, parent2):
    child1 = np.empty(11)
    child2 = np.empty(11)
    u = random.random() 
    n_c = 3 
    if (u < 0.5):
        beta = (2 * u)**((n_c + 1)**-1)
    else:
        beta = ((2*(1-u))**-1)**((n_c + 1)**-1)
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)
    child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)
    return child1, child2


def mutation(child):
    for i in range(CHROMOSOME_SIZE):
        mutation_prob = random.randint(0, 10)
        if mutation_prob < 3:
            if i <= 4:
                vary = 1 + random.uniform(-0.05, 0.05)
            else:
                vary = random.uniform(0, 1)
            rem = overfit_vector[i]*vary
            if abs(rem) <= 10:
                child[i] = rem
    return child


# survival of the fittest
def new_generation(parents_fitness, children, generation):
    children_fitness = calculate_fitness(children, generation+1)
    # print("Parents\n", parents_fitness)
    # print("Children\n", children_fitness)
    # parents_fitness = parents_fitness[:2]
    # children_fitness = children_fitness[:(POPULATION_SIZE-2)]
    generation = np.concatenate((parents_fitness, children_fitness))
    # print("\n\n", generation)
    # generation = generation[np.argsort(generation[:,-1])]
    # generation = parents_fitness + children_fitness
    generation = sorted(generation, key=lambda x:x[-1])
    # print("\n\n", generation)
    generation = generation[:POPULATION_SIZE]
    # print("\n\n", generation)
    return generation



# ------------------------------------------------ CHARLES DARWIN ----------------------------------------------------------------

first_population = initial_population()
population_fitness = calculate_fitness(first_population, 0)

for generation in range(GENERATIONS):
    # generate mating pool
    # print("\n\nGENERATION: ", generation)
    mating_pool = create_mating_pool(population_fitness, generation)
    # print("\n\nMating pool: \n", mating_pool)
    # breed children
    children = breed(mating_pool, generation)
    # print("Children: \n", children)
    # calculate fitness
    population_fitness = new_generation(mating_pool, children, generation)

# for generation in range(GENERATIONS):
#     print("\n\nGENERATION:", generation)
#     print("\n\nPARENTS:\n")
#     print(UNIVERSAL_DICT[generation])
#     print("\nMATING POOL:\n")
#     print(MATING_POOL[generation])
#     print("\nCHILDREN:\n")
#     print(CHILD_DICT[generation])

min_fitness = 1000000000000000000000000
gen = 0
for generation in range(GENERATIONS):
    for entry in UNIVERSAL_DICT[generation]:
        if entry["fitness"] < min_fitness:
            min_fitness = entry["fitness"]
            best_vec = np.array(entry["weights"])
            gen = generation

print("\n\nBest vector: ", list(best_vec))
print("\nFitness: ", min_fitness)
print("\nGeneration: ", gen)


SORTED_DATA = sorted(DATA_LIST, key = lambda i: i["fitness"])

# print(SORTED_DATA)

data = []
for i in range(10):
    data.append(SORTED_DATA[i]["weights"])

write_file("output.txt", list(data))

directory = "Generations"
parent_path = os.path.abspath(os.getcwd())
path = os.path.join(parent_path, directory)
os.mkdir(path)

for i in range(10):
    generation = SORTED_DATA[i]["generation"]
    print(generation)
    string = "generation_"+str(i+1)+".txt"
    file_path = os.path.join(path, string)
    with open(file_path, 'w') as f:
        if generation == 0:
            print("Population:", file = f)
            print(UNIVERSAL_DICT[generation], file = f)
            print("\nVector Details:", file = f)
            print(SORTED_DATA[i], file = f)
        elif generation == 10:
            for j in range(generation):
                print("Population for Generation ", j, file = f)
                print(UNIVERSAL_DICT[j], file = f)
                print("\n\nMating pool: ", file = f)
                print(MATING_POOL[j], file = f)
                print("\n\nAfter crossover and mutation:\n", file = f)
                print("\n", file = f)
            print("\nChild details:\n", file = f)
            for item in CHILD_DICT[generation-1]:
                if item["child"] == SORTED_DATA[i]["weights"]:
                    print(item, file = f)
            print(SORTED_DATA[i], file = f)
        else:
            for j in range(generation):
                print("Population for Generation ", j, file = f)
                print(UNIVERSAL_DICT[j], file = f)
                print("\n\nMating pool: ", file = f)
                print(MATING_POOL[j], file = f)
                print("\n\nAfter crossover and mutation:\n", file = f)
                print("\n", file = f)
            print("\nChild details:\n", file = f)
            for item in CHILD_DICT[generation-1]:
                if item["child"] == SORTED_DATA[i]["weights"]:
                    print(item, file = f)
            print(SORTED_DATA[i], file = f)
