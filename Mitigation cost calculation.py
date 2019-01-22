# -*- coding: utf8 -*-

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
import tqdm

import plotly.offline as pyo
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as pyt

import json_tricks
def exportOutput(outp, name):
    with open(name+'.json', 'w') as outfile:
        json_tricks.dump(outp,outfile)


##################
##
## Calculate sensitivity indices
##
##################

## Create task to be run in parallel
def parallelTask(i):
    T = Tvalues[i]
    numSamplesPerRun = 1000000
    numIdenticalRuns = 50 # Repeat the same calculation for increased accuracy
    return i, [
        sobol.testSensitivity(
            numSamplesPerRun,
            T,
            distributions.distributionsCosts,
            model.modelCosts,
            [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], # Second order indices. E.g., (0,1) is interaction between 1st and 2nd param
            (0,1,2) 
        ) for i in range(numIdenticalRuns)
    ]


## Define temperature values for which we want to calculate the carbon budget sensitivity
Tvalues = np.linspace(1.0, 5, 50)


## Calculate the sensitivity for each temperature, in parallel
pbar = tqdm.tqdm(total=len(Tvalues))
result = [None] * len(Tvalues)
def update(outp):
    i = outp[0]
    output = outp[1]
    result[i] = output
    pbar.update();

pool = multiprocessing.Pool()
for i in range(len(Tvalues)):
    pool.apply_async(parallelTask, (i,), callback=update)
pool.close()  
pool.join() 



exportOutput({'Tvalues': Tvalues, 'results': result}, 'relativeVariancesModel%i' % model.modelNum)


## Process the results such that they are better readable
processed = plot.processResults(result)


# processed[0] bevat first-order relative variance voor elke parameter (kolom)
# en voor elke temperatuur (rij)

# processed[1] bevat tweede orde termen
# processed[2] bevat derde orde term



##################
##
## Plot the partial variances
##
##################


namesOrig = ['TCRE', 'T₂₀₁₀', 'σ_nonCO₂', 'p']
desiredOrder = [0, 1, 3, 2]
firstOrderValues = processed[0][:,desiredOrder]
names = ['1st: '+namesOrig[i] for i in desiredOrder]

labels2nd = []
for i in range(len(namesOrig)):
    for j in range(i+1, len(namesOrig)):
        labels2nd += ['2nd: ' + namesOrig[i]+' ↔ '+namesOrig[j]]

withSecondOrder = np.concatenate([firstOrderValues, processed[1]], 1)
names += labels2nd

withThirdOrder = np.concatenate([withSecondOrder, processed[2]], 1)
names += ['3rd: TCRE ↔ T₂₀₁₀ ↔ σ_nonCO₂']

other_fill = np.maximum(0,1 - np.sum(withThirdOrder,1))
withOther = np.concatenate([withThirdOrder, np.transpose([other_fill])], 1)
names += ['Other']


colors = [
    'rgb(101,136,175)','rgb(136,191,186)','rgb(241,155,76)','rgb(225,108,111)',
    '#2ca02c','#17becf','#bcbd22',
    'rgb(101,136,175)','rgb(136,191,186)','rgb(241,155,76)','rgb(225,108,111)',
    'rgba(150,150,150,.5)'
]


fig = plot.cumulativeAreaChart(
    Tvalues, withOther, colors, names,
    arrowshift=[0,18,0,-12,-15,0,0,15,-12,-35,-60, -65], 
    textcolor=['#FFF']*12
)
pyo.plot(fig, image='', filename='relativeVarianceCostModel%i.html' % model.modelNum, image_width=fig.layout.width, image_height=fig.layout.height, show_link=False)