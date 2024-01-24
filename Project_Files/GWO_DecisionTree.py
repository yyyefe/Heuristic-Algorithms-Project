import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
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
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': np.arange(1, 101),
        'min_samples_split': np.arange(2, 101),
        'min_samples_leaf': np.arange(1, 101),
        }
    population_size = 20
    num_iterations = 50
    
    AlphaScore, best_hyperparameters, best_accuracy = GWO(
        population_size, num_iterations, param_grid,
        Xtrain_scaled, ytrain, Xtest_scaled, ytest
        )
    
    #print("Best Parameters:", best_hyperparameters)
    #print("Best Accuracy:", best_accuracy)
    
    return AlphaScore, best_hyperparameters, best_accuracy

def initialize_population(population_size, param_grid):
    population = []

    for _ in range(population_size):
        hyperparameters = {}
        for param_name, param_values in param_grid.items():
            hyperparameters[param_name] = np.random.choice(param_values)

        population.append(hyperparameters)
    
    return population

def fitness_function(hyperparameters, Xtrain_scaled, ytrain, Xtest_scaled, ytest):
    tree = DecisionTreeClassifier(
        criterion=hyperparameters['criterion'],
        splitter=hyperparameters['splitter'],
        max_depth=int(hyperparameters['max_depth']),
        min_samples_split=int(hyperparameters['min_samples_split']),
        min_samples_leaf=int(hyperparameters['min_samples_leaf']),
        random_state=122
    )
    tree.fit(Xtrain_scaled, ytrain)
    y_pred = tree.predict(Xtest_scaled)
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    f1 = f1_score(ytest, y_pred)
    return accuracy


def GWO(SearchAgents_no, Max_iter, param_grid, Xtrain_scaled, ytrain, Xtest_scaled, ytest):

    # initialize alpha, beta, and delta_pos
    Alpha_pos = initialize_population(1, param_grid)
    Alpha_score = 0

    Beta_pos = initialize_population(1, param_grid)
    Beta_score = 0

    Delta_pos = initialize_population(1, param_grid)
    Delta_score = 0
    
    # Initialize the positions of search agents
    Positions = initialize_population(SearchAgents_no, param_grid)
    
    AlphaScore = []

    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):
            # Calculate objective function for each search agent
            fitness = fitness_function(Positions[i], Xtrain_scaled, ytrain, Xtest_scaled, ytest)
            # Update Alpha, Beta, and Delta
            if fitness > Alpha_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i].copy()

            if fitness < Alpha_score and fitness > Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i].copy()

            if fitness < Alpha_score and fitness < Beta_score and fitness > Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i].copy()

        AlphaScore.append(Alpha_score)
        #print(Alpha_pos)
        #print(Alpha_score)
        
        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            r1 = random.random()  # r1 is a random number in [0,1]
            r2 = random.random()  # r2 is a random number in [0,1]

            A1 = 2 * a * r1 - a
            # Equation (3.3)
            C1 = 2 * r2
            # Equation (3.4)

            D_alpha = abs(C1 * Alpha_pos['max_depth'] - Positions[i]['max_depth'])
            # Equation (3.5)-part 1
            X1_1 = Alpha_pos['max_depth'] - A1 * D_alpha
            # Equation (3.6)-part 1
            D_alpha = abs(C1 * Alpha_pos['min_samples_split'] - Positions[i]['min_samples_split'])
            # Equation (3.5)-part 1
            X1_2 = Alpha_pos['min_samples_split'] - A1 * D_alpha
            # Equation (3.6)-part 1
            D_alpha = abs(C1 * Alpha_pos['min_samples_leaf'] - Positions[i]['min_samples_leaf'])
            # Equation (3.5)-part 1
            X1_3 = Alpha_pos['min_samples_leaf'] - A1 * D_alpha
            # Equation (3.6)-part 1

            r1 = random.random()
            r2 = random.random()

            A2 = 2 * a * r1 - a
            # Equation (3.3)
            C2 = 2 * r2
            # Equation (3.4)

            D_beta = abs(C2 * Beta_pos['max_depth'] - Positions[i]['max_depth'])
            # Equation (3.5)-part 2
            X2_1 = Beta_pos['max_depth'] - A2 * D_beta
            # Equation (3.6)-part 2
            D_beta = abs(C2 * Beta_pos['min_samples_split'] - Positions[i]['min_samples_split'])
            # Equation (3.5)-part 2
            X2_2 = Beta_pos['min_samples_split'] - A2 * D_beta
            # Equation (3.6)-part 2
            D_beta = abs(C2 * Beta_pos['min_samples_leaf'] - Positions[i]['min_samples_leaf'])
            # Equation (3.5)-part 2
            X2_3 = Beta_pos['min_samples_leaf'] - A2 * D_beta
            # Equation (3.6)-part 2

            r1 = random.random()
            r2 = random.random()

            A3 = 2 * a * r1 - a
            # Equation (3.3)
            C3 = 2 * r2
            # Equation (3.4)

            D_delta = abs(C3 * Delta_pos['max_depth'] - Positions[i]['max_depth'])
            # Equation (3.5)-part 3
            X3_1 = Delta_pos['max_depth'] - A3 * D_delta
            # Equation (3.5)-part 3
            D_delta = abs(C3 * Delta_pos['min_samples_split'] - Positions[i]['min_samples_split'])
            # Equation (3.5)-part 3
            X3_2 = Delta_pos['min_samples_split'] - A3 * D_delta
            # Equation (3.5)-part 3
            D_delta = abs(C3 * Delta_pos['min_samples_leaf'] - Positions[i]['min_samples_leaf'])
            # Equation (3.5)-part 3
            X3_3 = Delta_pos['min_samples_leaf'] - A3 * D_delta
            # Equation (3.5)-part 3

            if (X1_1 + X2_1 + X3_1) / 3 > 1:
                Positions[i]['max_depth'] = int((X1_1 + X2_1 + X3_1) / 3)  # Equation (3.7)
            if (X1_2 + X2_2 + X3_2) / 3 > 2:
                Positions[i]['min_samples_split'] = int((X1_2 + X2_2 + X3_2) / 3)  # Equation (3.7)
            if (X1_3 + X2_3 + X3_3) / 3 > 1:
                Positions[i]['min_samples_leaf'] = int((X1_3 + X2_3 + X3_3) / 3)  # Equation (3.7)

    return AlphaScore, Alpha_pos, Alpha_score
