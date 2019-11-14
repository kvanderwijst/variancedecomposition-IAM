###################
##
## Preliminaries
##
###################

## Import packages

import numpy as np
import scipy.stats as stats
from variancedecomposition.model import modelNum


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

def sample_lognormal(num, mu, sigma):
    return np.random.lognormal(mean=mu, sigma=sigma, size=num)

def sample_trunclognorm(num, pStar, pStar_sigma, low=0.0, high=1.5):
    dist = stats.lognorm(s=pStar_sigma, scale=pStar)
    
    CDF_low = dist.cdf(low)
    CDF_high = dist.cdf(high)
    
    unifsample = (CDF_high - CDF_low) * np.random.random(num) + CDF_low
    
    return dist.ppf(unifsample)

######## Distribution for full mitigation cost model
#pStar = 0.24154
#pStar = 0.109
pStar = 0.20615
#pStar_sigma = 1.08
pStar_sigma = 0.83555

#distribution_p = lambda N: sample_betaPERT(N, 0, pStar, 1)
#distribution_p = lambda N: sample_lognormal(N, np.log(pStar), pStar_sigma)
distribution_p = lambda N: sample_trunclognorm(N, pStar, pStar_sigma, high=1.5)

sigma_nonCO2_std = 0.121

if modelNum == 0:
# TCRE from pink plume, symmetrical distribution (Gaussian)

    distributionsCosts = [
        lambda N: sample_normal(N, 0.62, 0.12), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std),  # sigma_nonCO2
        distribution_p   # p
    ]

elif modelNum == 1:
# TCRE from pink plume, asymmetrical distribution (beta-PERT)

    distributionsCosts = [
        lambda N: sample_betaPERT(N, 0.255, 0.62, 0.855), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std),  # sigma_nonCO2
        distribution_p   # p
    ]

elif modelNum == 2:
# TCRE from gray plume , linear non-CO2

    distributionsCosts = [
        lambda N: sample_normal(N, 0.45, 0.12), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std),  # sigma_nonCO2
        distribution_p   # p
    ]

elif modelNum == 3:
# TCRE from gray plume , convex non-CO2

    distributionsCosts = [
        lambda N: sample_normal(N, 0.45, 0.12), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std / 0.60713),  # sigma_nonCO2 
                    # is different from others, since it is still multiplied
                    # by forcingToTemperature term
        distribution_p   # p
    ]

elif modelNum == 3.1:
# TCRE from gray plume , convex non-CO2

    distributionsCosts = [
        lambda N: sample_normal(N, 0.45, 0.12), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std / 0.60713 / 2),  # sigma_nonCO2 
                    # is different from others, since it is still multiplied
                    # by forcingToTemperature term
                    # However, to obtain pink plume, twice the MAGICC coefficient is used
                    # This leads to an overestimation of 2 of the uncertainty. This is dealt
                    # with here.
        distribution_p   # p
    ]

elif modelNum == 4:
# TCRE from Collins et al (2013) , linear non-CO2

    distributionsCosts = [
        lambda N: sample_normal(N, 0.45, 0.25), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std),  # sigma_nonCO2
        distribution_p   # p
    ]

elif modelNum == 5:
	sigma_logprice = 0.85114

	distribution_p = lambda N: sample_normal(N, 0, sigma_logprice)

	distributionsCosts = [
        lambda N: sample_betaPERT(N, 0.255, 0.62, 0.855), # TCRE
        lambda N: sample_normal(N, 0.909, 0.15/2), # T2010
        lambda N: sample_normal(N, 0, sigma_nonCO2_std),  # sigma_nonCO2
        distribution_p   # p
    ]

######## Distribution for carbon budget model

# The calculation of the carbon budget uses the same
# parameter distribution as the full model, without
# the cost parameter p
distributionsCarbonBudget = distributionsCosts[:3]


