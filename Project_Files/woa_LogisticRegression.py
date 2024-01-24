import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
import random
import copy

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
        'penalty': ['l1', 'l2'],
        'C':np.arange(0.0001, 2, 0.0001),
        'solver': ['liblinear', 'saga']
    }
    population_size = 100
    num_iterations = 100

    leaderFitness, best_hyperparameters, best_accuracy = whale_optimization_algorithm(
        population_size, num_iterations, param_grid,
        Xtrain_scaled, ytrain, Xtest_scaled, ytest
    )

    #print("Best Parameters:", best_hyperparameters)
    #print("Best Accuracy:", best_accuracy)
    
    return leaderFitness, best_hyperparameters, best_accuracy

def initialize_population(population_size, param_grid):
    population = []

    for _ in range(population_size):
        hyperparameters = {}
        for param_name, param_values in param_grid.items():
            hyperparameters[param_name] = np.random.choice(param_values)

        population.append(hyperparameters)
    
    return population

def fitness_function(hyperparameters, Xtrain_scaled, ytrain, Xtest_scaled, ytest):
    
    logreg = LogisticRegression(
        penalty=hyperparameters['penalty'],
        solver=hyperparameters['solver'],
        C=hyperparameters['C'],
        random_state=122, 
        max_iter=5000
    )
    logreg.fit(Xtrain_scaled, ytrain)
    y_pred = logreg.predict(Xtest_scaled)
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    return accuracy


def whale_optimization_algorithm(population_size, num_iterations, param_grid, Xtrain_scaled, ytrain, Xtest_scaled, ytest):
    population = initialize_population(population_size, param_grid)
    leader_position = population[np.random.randint(0, len(population))]
    leader_fitness = fitness_function(leader_position, Xtrain_scaled, ytrain, Xtest_scaled, ytest)
    leaderFitness = []
    #print(leader_position)
    #print(leader_fitness)
    
    for iteration in range(num_iterations):
        rnd = random.Random(0)    
        fitness = []     
        for i in range(population_size):
            fitness.append(fitness_function(population[i], Xtrain_scaled, ytrain, Xtest_scaled, ytest))
            if fitness[i] > leader_fitness:
                leader_fitness = fitness[i]
                leader_position = population[i]
                leader = copy.deepcopy(population[i])
                #print(leader)
                #print(leader_fitness)
    
        leaderFitness.append(leader_fitness)   
        
        # linearly decreased from 2 to 0
        a = 2 * (1 - iteration / num_iterations)
        a2=-1+iteration*((-1)/num_iterations) 
        
        for i in range(population_size):
            A = 2 * a * rnd.random() - a
            C = 2 * rnd.random()
            b = 1
            l = (a2-1)*rnd.random()+1
            p = rnd.random()
            if p < 0.5:
                if abs(A) < 1:
                    new_position = population[i]
                    D = abs(C * leader_position['C'] - population[i]['C'])
                    if leader_position['C'] - A * D > 0:
                        new_position['C'] = leader_position['C'] - A * D
                    
                elif abs(A) > 1:
                    random_whale_index = rnd.randint(0, population_size - 1)
                    random_whale_position = population[random_whale_index]
                    new_position = random_whale_position
                    D = abs(C * random_whale_position['C'] - population[i]['C'])
                    if random_whale_position['C'] - A * D > 0:
                        new_position['C'] = random_whale_position['C'] - A * D
                    
            
            elif p >= 0.5:
                new_position = population[i]
                D = abs(leader_position['C'] - population[i]['C'])
                if D * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_position['C'] > 0:
                    new_position['C'] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_position['C']

            # Check if any search agent goes beyond the search space and amend it
            population[i] = new_position
            
    return leaderFitness, leader, leader_fitness
