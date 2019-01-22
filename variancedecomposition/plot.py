## Import packages

import plotly.offline as pyo
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as pyt

import datetime
import numpy as np



####### First process results obtained from sobol method to a more readable format

def processResults (results):
    allFirstOrders = []
    allSecondOrders = []
    allThirdOrders = []
    
    # For every temperature value:
    for Ti in range(len(results)):
        res = results[Ti]
        
        firstOrders = []
        secondOrders = []
        thirdOrders = []
        
        # For every repeated sampling
        for j in range(len(res)):
            firstOrders += [res[j][0]]
            secondOrders += [res[j][1]]
            thirdOrders += [res[j][2]]
        
        #print(firstOrders)
        
        # Average all variance estimators
        allFirstOrders += [np.mean(firstOrders,0)]
        allSecondOrders += [np.mean(secondOrders,0)]
        allThirdOrders += [np.mean(thirdOrders,0)]
    
    return (
        np.maximum(0,np.array(allFirstOrders)),
        np.maximum(0,np.array(allSecondOrders)),
        np.maximum(0,np.array(allThirdOrders))
    )




######## Create a cumulative area chart

def cumulativeAreaChart (xvalues, yvalues, colors, names, arrowshift=None, textcolor=None):
    
    # Calculate cumulative values for stacked chart
    cumulative = [
        np.sum(yvalues[:,0:i+1],1)
        for i in range(yvalues.shape[1])
    ]

    # Plot
    traces = [
        go.Scatter(
            x=xvalues, y=cumulative[i],
            #name=names[i],
            mode='lines',
            fill='tonexty',
            fillcolor=colors[i % len(colors)],
            text=[str(int(y*100))+'%' for y in yvalues[:,i]],
            hoverinfo='x+text'
        )
        for i in range(yvalues.shape[1])]
    
    if arrowshift == None:
        arrowshift = np.zeros(yvalues.shape[1])
    if textcolor == None:
        textcolor = ['#000'] * yvalues.shape[1]
    
    layout = go.Layout(
        colorway=colors,
        barmode='stack',
        width=900,
        height=900,
        xaxis=dict(title='Temperature change (°C)', range=[1.5,5]),
        yaxis=dict(title='Rel. contribution to variance'),
        font = dict(size=13),
        margin=dict(r=225, t=30,l=60),
        annotations=[
            dict(
                x=np.max(xvalues), y=cumulative[i][-1] - yvalues[-1,i]/2,
                xref='x', yref='y',
                xanchor='left',
                text=names[i],
                showarrow=True,
                font=dict(color=textcolor[i], size=14),
                arrowhead=1, arrowsize=1, arrowwidth=2,
                arrowcolor=colors[i % len(colors)], ax=20, ay=arrowshift[i],
                borderwidth=0, bgcolor=colors[i % len(colors)]
            ) for i in range(len(names))
        ],
        showlegend=False
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig