# Variance decomposition of simple Integrated Assessment Model

## The IAM
First, we calculate a carbon budget given a temperature goal, as function of three parameters: climate sensitivity (TCRE), temperature in 2010 and influence of non-CO2 gases.

Given the mitigation costs as function of carbon budget, and given the carbon budget calculation from the first part, we calculate the costs as function of temperature goal.

## Variance decomposition
Using Sobol's Monte Carlo method, we calculate the contribution to the variance of each input parameter.

## Usage
To calculate the partial variances of the carbon budget only, run Carbon budget calculation.py

To calculate the partial variances of the full model, run Mitigation cost calculation.py


The underlying distributions can be edited in variancedecomposition/distributions.py, specifically in the distributionCosts variable. If the model functions need to be changed, the variancedecomposition/model.py is the place to go. This also contained the exponential functions used to model mitigation costs.

