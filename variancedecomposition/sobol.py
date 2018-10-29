###################
##
## Variance decomposition
## using Sobol's Monte Carlo method
##
###################

import numpy as np

####### Create the sample matrices
## These are the matrices A and B, 
## and matrix created as combination between A and B
## by keeping certain columns (parameters) fixed


def createSampleMatrices (N, T, distributions, model, secondOrder, thirdOrder):
    """
    T: temperature level between 0 and 5
    """
    
    k = len(distributions)
    
    samples = [dist(2*N) for dist in distributions]
    
    A = np.transpose(np.array([sample[:N] for sample in samples]))
    B = np.transpose(np.array([sample[N:] for sample in samples]))
    
    C = [None]*k
    for i in range(k):
        C[i] = np.copy(B)
        C[i][:,i] = A[:,i]
    
    # Create a second order matrix
    D = [None]*len(secondOrder)
    for i in range(len(secondOrder)):
        D[i] = np.copy(B)
        D[i][:,secondOrder[i][0]] = A[:,secondOrder[i][0]];
        D[i][:,secondOrder[i][1]] = A[:,secondOrder[i][1]];
    
    # And a third order matrix
    if thirdOrder is not False:
        E = np.copy(B)
        E[:,thirdOrder[0]] = A[:,thirdOrder[0]];
        E[:,thirdOrder[1]] = A[:,thirdOrder[1]];
        E[:,thirdOrder[2]] = A[:,thirdOrder[2]];

        yE = model(T, E)
    else:
        E = []
        yE = np.array([0])
    
    yA = model(T, A)
    yB = model(T, B)
    
    yC = [model(T, C[i]) for i in range(k)]
    
    yD = [model(T, D[i]) for i in range(len(secondOrder))]
    
    del(A, B, C, D, E)
    
    return (yA, yB, yC, yD, yE)



####### Create the variance estimators
## Either the nth-order estimator or the total variance

def nthOrderEstimators (yA, yB, yC_set, calctotal = False):
    # yA and yB are the resulting model values calculating
    # using two fully resampled parameter values
    #
    # yC_set are the model values using parameter values
    # from matrix B, with a chosen set of columns taken from
    # matrix A.
    N = len(yA)
    
    f0square = np.mean(yA)**2
    
    VY = np.sum(yA * yA) / N - f0square
    
    # Estimation of S_i
    firstorder = (np.sum(yA * yC_set) / N - f0square) / VY
    
    # Estimation of S_Ti
    if calctotal:
        totaleffect = 1 - (np.sum(yB * yC_set) / N - f0square) / VY
        return (firstorder, totaleffect)
    else:
        return firstorder




####### Combine the creation of parameter values and the calculation
####### of sensitivity 

def testSensitivity (N, T, distributions, model, secondOrderIndices, thirdOrderIndices):
    # Create model values using parameter distributions
    yA, yB, yC, yD, yE = createSampleMatrices(N, T, distributions, model, secondOrderIndices, thirdOrderIndices)

    k = len(distributions)

    # Calculate all first order variance terms
    firstOrders = [nthOrderEstimators(yA, yB, yC[i]) for i in range(k)]

    # Calculate the second order variance terms defined in the input parameter secondOrderIndices
    secondOrders = [nthOrderEstimators(yA, yB, yD[i]) - firstOrders[secondOrderIndices[i][0]] - firstOrders[secondOrderIndices[i][1]] for i in range(len(secondOrderIndices))]

    # Also calculate the third order term if necessary
    if thirdOrderIndices is not False:
        thirdOrders = [nthOrderEstimators(yA, yB, yE) - np.sum(secondOrders) - firstOrders[0] - firstOrders[1] - firstOrders[2]]
    else:
        thirdOrders = []

    del(yA, yB, yC, yD, yE)
    return [firstOrders, secondOrders, thirdOrders]

