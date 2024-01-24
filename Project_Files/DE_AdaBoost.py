import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import random
import copy
import math
import time

def baslat():
    
    data = pd.read_csv('heart.csv')
    
    train, test = train_test_split(data, test_size=0.2, random_state=122)
    
    Xtrain = train.drop(columns=['target'], axis=1)
    ytrain = train['target']
    
    Xtest = test.drop(columns=['target'], axis=1)
    ytest = test['target']
    
    scaler = MinMaxScaler()
    Xtrain_scaled = scaler.fit_transform(Xtrain)
    Xtest_scaled = scaler.transform(Xtest)
    
    # Hyperparameter grid'ini tanımlayın
    param_grid = {
        'n_estimators': np.arange(1, 501),
        'learning_rate': np.arange(0.01, 1, 0.01),
        'algorithm': ['SAMME', 'SAMME.R']
        }
    
    population_size = 10
    num_iterations = 10
    
    bestFitness, best_hyperparameters, best_accuracy = DE(
        population_size, num_iterations, param_grid,
        Xtrain_scaled, ytrain, Xtest_scaled, ytest
        )
    
    #print("Best Parameters:", best_hyperparameters)
    #print("Best Accuracy:", best_accuracy)
    
    return bestFitness, best_hyperparameters, best_accuracy

def initialize_population(population_size, param_grid):
    population = []

    for _ in range(population_size):
        hyperparameters = {}
        for param_name, param_values in param_grid.items():
            hyperparameters[param_name] = np.random.choice(param_values)

        population.append(hyperparameters)
    
    return population

def fitness_function(hyperparameters, Xtrain_scaled, ytrain, Xtest_scaled, ytest):
    AdaBoost = AdaBoostClassifier(
        n_estimators=int(hyperparameters['n_estimators']),
        learning_rate=hyperparameters['learning_rate'],
        algorithm=hyperparameters['algorithm'],
        random_state=122
    )
    AdaBoost.fit(Xtrain_scaled, ytrain)
    y_pred = AdaBoost.predict(Xtest_scaled)
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    return accuracy


# Differential Evolution (DE)
# mutation factor = [0.5, 2]
# crossover_ratio = [0,1]
def DE(population_size, num_iterations, param_grid, Xtrain_scaled, ytrain, Xtest_scaled, ytest):

    mutation_factor = 0.5
    crossover_ratio = 0.7
    stopping_func = None
    bestFitness = []

    # initialize population
    population = initialize_population(population_size, param_grid)

    # calculate fitness for all the population
    population_fitness = [] 
    best = 0
    for i in range(population_size):
        population_fitness.append(fitness_function(population[i], Xtrain_scaled, ytrain, Xtest_scaled, ytest))
        if population_fitness[i] > best:
            best = population_fitness[i]
            leader_solution = population[i]
            #print(leader_solution)
            #print(best)

    t = 0
    while t < num_iterations:
        # should i stop
        if stopping_func is not None and stopping_func(best, leader_solution, t):
            break
           
        bestFitness.append(best)

        # loop through population
        for i in range(population_size):
            # 1. Mutation

            # select 3 random solution except current solution
            ids_except_current = [_ for _ in range(population_size) if _ != i]
            id_1, id_2, id_3 = random.sample(ids_except_current, 3)

            mutant_sol = []
            
            mutant_sol = copy.deepcopy(population[id_1])
            if population[id_1]['n_estimators'] + mutation_factor * (population[id_2]['n_estimators'] - population[id_3]['n_estimators']) > 1:
                mutant_sol['n_estimators'] = int(population[id_1]['n_estimators'] + mutation_factor * (population[id_2]['n_estimators'] - population[id_3]['n_estimators']))
            if population[id_1]['learning_rate'] + mutation_factor * (population[id_2]['learning_rate'] - population[id_3]['learning_rate']) > 0:
                mutant_sol['learning_rate'] = population[id_1]['learning_rate'] + mutation_factor * (population[id_2]['learning_rate'] - population[id_3]['learning_rate'])
           
            # 2. Recombination
            rn = random.uniform(0, 1)
            if rn  > crossover_ratio:
                mutant_sol = population[i]

            # 3. Replacement / Evaluation

            # calc fitness
            mutant_fitness = fitness_function(mutant_sol, Xtrain_scaled, ytrain, Xtest_scaled, ytest)
            # s.func_evals += 1

            # replace if mutant_fitness is better
            if mutant_fitness > population_fitness[i]:
                population[i] = mutant_sol
                population_fitness[i] = mutant_fitness

                # update leader
                if mutant_fitness > best:
                    best = mutant_fitness
                    leader_solution = mutant_sol 
                    #print(leader_solution)
                    #print(best)

        # increase iterations
        t = t + 1

    # return solution
    return bestFitness, leader_solution, best