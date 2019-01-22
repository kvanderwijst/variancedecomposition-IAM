###################
##
## Calculate carbon budget sensitivity
##
###################

## Import packages
import numpy as np
import variancedecomposition.distributions as distributions
import variancedecomposition.sobol as sobol
import variancedecomposition.model as model
import variancedecomposition.plot as plot
import multiprocessing

import plotly.offline as pyo
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as pyt


##################
##
## Calculate sensitivity indices
##
##################


## Create task to be run in parallel
def parallelTask(T):
    numSamplesPerRun = 1000000
    numIdenticalRuns = 50 # Repeat the same calculation for increased accuracy
    return [
        sobol.testSensitivity(
            numSamplesPerRun,
            T,
            distributions.distributionsCarbonBudget,
            model.modelCarbonBudget,
            [(0,1),(0,2),(1,2)], # Second order indices. E.g., (0,1) is interaction between 1st and 2nd param
            False # Do not calculate third order terms
        ) for i in range(numIdenticalRuns)
    ]


## Define temperature values for which we want to calculate the carbon budget sensitivity
Tvalues = np.linspace(1.0, 5, 50)


## Calculate the sensitivity for each temperature, in parallel
pool = multiprocessing.Pool()
result = pool.map(parallelTask, Tvalues)
pool.close()  
pool.join() 


## Process the results such that they are better readable
processed = plot.processResults(result)

# processed[0] bevat first-order relative variance voor elke parameter (kolom)
# en voor elke temperatuur (rij)

# processed[1] bevat tweede orde termen



##################
##
## Plot the partial variances
##
##################

# We only look at first order variances here
firstOrderValues = processed[0]

# Calculate the rest (interaction terms)
other_fill = np.maximum(0,1 - np.sum(firstOrderValues,1))
withOther = np.concatenate([firstOrderValues, np.transpose([other_fill])], 1)

fig = plot.cumulativeAreaChart(
    xvalues=Tvalues,
    yvalues=withOther,
    colors=['rgb(101,136,175)','rgb(136,191,186)','rgb(241,155,76)','rgba(150,150,150,.5)'],
    names=['TCRE', 'T2010', 'sigma_nonCO2', 'Interaction terms'],
    textcolor=['#FFF']*4
)
fig['layout'].update(
    xaxis=dict(range=[np.min(Tvalues),np.max(Tvalues)]),
    yaxis=dict(tickformat='%'),
    margin=dict(r=170,t=10),
    height=500,
    width=800
)
pyo.plot(fig, image_width=fig.layout.width, image_height=fig.layout.height, show_link=False)