###################
##
## Preliminaries
##
###################

## Import packages

import numpy as np
import scipy.stats as stats
import variancedecomposition.model as model


######## Define the sampling functions

## Beta-PERT distribution

def betaPERT_PDF (x, a, b, c):
    alpha = (4*b+c-5*a)/(c-a)
    beta = (5*c-a-4*b)/(c-a)
    z = (x-a)/(c-a)
    return stats.beta.pdf(z, alpha, beta)

def betaPERT_CDF (x, a, b, c):
    alpha = (4*b+c-5*a)/(c-a)
    beta = (5*c-a-4*b)/(c-a)
    z = (x-a)/(c-a)
    return stats.beta.cdf(z, alpha, beta)

def betaPERT_iCDF (y, a, b, c):
    alpha = (4*b+c-5*a)/(c-a)
    beta = (5*c-a-4*b)/(c-a)
    z = stats.beta.ppf(y, alpha, beta)
    return z * (c-a) + a

def sample_betaPERT(num, a, b, c):
    # First draw uniformly distributed numbers between 0 and 1
    unifsample = np.random.random(num)
    
    # Then transform these to desired distribution using inverse CDF transform
    return betaPERT_iCDF(unifsample, a, b, c)



######## Normal distribution

def sample_normal(num, mu, sigma):
    return sigma * np.random.randn(num) + mu



######## Distribution for full mitigation cost model
pStar = 0.24154
distributionsCosts = [
    lambda N: sample_betaPERT(N, 0.255, 0.62, 0.855), # TCRE
    lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
    lambda N: sample_normal(N, 0, .135),  # sigma_nonCO2
    lambda N: sample_betaPERT(N, 0, pStar, 1)   # p
]
distributionsMode = [0.62, 1, 0, pStar]

######## Distribution for carbon budget model

# The calculation of the carbon budget uses the same
# parameter distribution as the full model, without
# the cost parameter p
distributionsCarbonBudget = distributionsCosts[:3]


