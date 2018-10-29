###################
##
## Define the model used
##
## Here: carbon budget calculation 
## and mitigation cost as function of temperature
##
###################



####### Define individual components
####### ----------------------------


## Carbon budget

def CO2asfunctionofTemperature (T, TCRE, T0, sigma_nonCO2):
    return (T - T0 - sigma_nonCO2) / TCRE



## Mitigation cost functions

def expfct(CO2, a, b):
    value = a * np.exp(-b*CO2) - a*np.exp(-5.5*b)
    return np.where(CO2 < 5.5, value, 0)

def fMax(CO2):
    return expfct(CO2, 14.9766, 0.541945)

def fMin(CO2):
    return expfct(CO2, 0.886887, 3.47027)

def f(CO2, p):
    return fMin(CO2) + (fMax(CO2) - fMin(CO2)) * p

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



def modelCosts(T, paramvalues):
    TCRE = paramvalues[:,0]
    T2010 = paramvalues[:,1]
    sigma_nonCO2 = paramvalues[:,2]
    p = paramvalues[:,3]
    return f(CO2asfunctionofTemperature(T, TCRE, T2010, sigma_nonCO2), p)