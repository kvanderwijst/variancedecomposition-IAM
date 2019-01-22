###################
##
## Define the model used
##
## Here: carbon budget calculation 
## and mitigation cost as function of temperature
##
###################

import numpy as np


#### DEFINE MODEL
##
## 0: TCRE from pink plume, symmetrical distribution
## 1: TCRE from pink plume, asymmetrical (Beta-PERT) distribution
## 2: TCRE from gray plume + linear non-CO2 contributions
## 3: TCRE from gray plume + convex non-CO2 contributions
## 4: TCRE from Collins et al (2013) + linear non-CO2 contributions

modelNum = 0



####### Define individual components
####### ----------------------------


## Carbon budget

if modelNum in [0, 1]:
# TCRE from pink plume

    def CO2asfunctionofTemperature (T, TCRE, T0, sigma_nonCO2):
        return (T - T0 - sigma_nonCO2) / np.maximum(0.001, TCRE)

    def TemperatureasfunctionofCO2 (CO2, TCRE, T0, sigma_nonCO2):
        return T0 + TCRE * CO2 + sigma_nonCO2

elif modelNum in [2, 4]:
# TCRE from gray plume or from Collins + linear non-CO2

    TCRE_nonCO2 = 0.17
    
    def CO2asfunctionofTemperature (T, TCRE_CO2, T0, sigma_nonCO2):
        return (T - T0 - sigma_nonCO2) / np.maximum(0.001, TCRE_CO2 + TCRE_nonCO2)
    
    def TemperatureasfunctionofCO2 (CO2, TCRE_CO2, T0, sigma_nonCO2):
        return T0 + (TCRE_CO2 + TCRE_nonCO2) * CO2 + sigma_nonCO2
    
elif modelNum is 3:
# TCRE from gray plume + convex non-CO2:
    
    forcingToTemp = 2 * 0.60126 # Twice the MAGICC coefficient
    b = 0.0849
    c = 0.0083
    
    def CO2asfunctionofTemperature (T, TCRE_CO2, T0, sigma_nonCO2):
        return -(1.0/(2*c*forcingToTemp))*(b*forcingToTemp + TCRE - np.sqrt(-4*c*forcingToTemp*(forcingToTemp*sigma_nonCO2 - T + T0) + (b*forcingToTemp + TCRE)**2))
    
    def TemperatureasfunctionofCO2 (CO2, TCRE_CO2, T0, sigma_nonCO2):
        return T0 + TCRE_CO2 * CO2 + forcingToTemp * (b * CO2 + c * CO2**2 + sigma_nonCO2)



## Mitigation cost functions

def expfct(CO2, a, b):
    value = a * np.exp(-b*CO2) - a*np.exp(-5.5*b)
    return np.where(CO2 < 5.5, value, 0)

def fMax(CO2):
    return expfct(CO2, 14.9766, 0.541945)

def fMin(CO2):
    return expfct(CO2, 0.886887, 3.47027)

def f(CO2, p):
    safe_CO2 = np.maximum(-3, CO2)
    return fMin(safe_CO2) + (fMax(safe_CO2) - fMin(safe_CO2)) * p

# Translate indexed costs to USD
def toRealCosts(indexedCosts):
    return indexedCosts * 33.94462214705881 ## Is defined more clearly in thesis




####### Define model functions
####### ----------------------
#######
####### These functions take as input a chosen (deterministic) parameter
####### and a matrix of parameter values drawn from their respective 
####### distributions


def modelCarbonBudget(T, paramvalues):
    TCRE = paramvalues[:,0]
    T2010 = paramvalues[:,1]
    sigma_nonCO2 = paramvalues[:,2]
    return CO2asfunctionofTemperature(T, TCRE, T2010, sigma_nonCO2)

def sampleCO2value (T, N, distributions, CO2asfunctionofTemperature=CO2asfunctionofTemperature):
    # Sample data points
    sample = dict(
        TCRE = distributions[0](N),
        T2010 = distributions[1](N),
        sigma_nonCO2 = distributions[2](N)
    )

    # CO2 values
    CO2values = CO2asfunctionofTemperature(T, sample['TCRE'], sample['T2010'], sample['sigma_nonCO2'])
    
    return CO2values


def modelCosts(T, paramvalues):
    TCRE = paramvalues[:,0]
    T2010 = paramvalues[:,1]
    sigma_nonCO2 = paramvalues[:,2]
    p = paramvalues[:,3]
    return f(CO2asfunctionofTemperature(T, TCRE, T2010, sigma_nonCO2), p)