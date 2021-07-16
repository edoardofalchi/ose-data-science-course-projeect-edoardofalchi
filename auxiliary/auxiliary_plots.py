"""This module contains auxiliary functions for plots which are used in the main notebook."""

#%matplotlib inline
import plotly.express as px
import scipy
import numpy as np
from numpy import nan
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib as plt
import matplotlib.pyplot as pltpy
from IPython.display import HTML
import math



from auxiliary.auxiliary_prepare_data import *
from auxiliary.auxiliary_tables import *

def USA_geo(df):
    '''
    argument:dataset
    return:geographical spot plot
    '''
    fig = px.choropleth(locations=["NJ", "PA"], locationmode="USA-states",
                    color=df[df["time"]==0].groupby("state")["FTE"].mean(),
                    scope="usa", labels={"color":"FTE pre-intervention"}).update_layout(title="Where are the protagonists of our story located?")

    return fig.show()



def did_visual(NJ_before,NJ_after,PA_before,PA_after,NJ_counterfactual):
      

    
    did_pic= pltpy.figure(num=None, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
    fig, ax = pltpy.subplots()
    lineNJ, = ax.plot(['0', '1'], [NJ_before, NJ_after],color='blue',label='NJ before and after')
    linePA, = ax.plot(['0', '1'], [PA_before, PA_after],color = 'red',label = 'PA before and after')
    lineNJ0, = ax.plot(['0', '1'], [NJ_before, NJ_counterfactual],color = 'blue',linestyle='dashed',label='NJ counterfactual')
    ax.legend()
    pltpy.ylim(15, 28)
    plt.pyplot.axvline(x=0.5, color='black')
    plt.pyplot.xticks([0,0.5,1], ['before','policy into force','after'])
    plt.pyplot.xlabel('TIME')
    plt.pyplot.ylabel('EMPLOYMENT')
    pltpy.title("Difference-in-difference: Before and After", fontsize="14")
    return did_pic


def figure1(df_before,df_after):
    
    '''
    replication of Figure 1
    
    argument:subset for pre and post intervention
    return:distribution of starting wage rates
    '''
    plt.pyplot.figure(figsize=(15, 10), dpi=70, facecolor='w', edgecolor='k')
    plt.pyplot.subplots_adjust(wspace=0.2, hspace=0.4)

    plt.pyplot.xlabel('wage range')
    plt.pyplot.ylabel('percent of stores $*(10^{-1})$')
    plt.pyplot.subplot(1, 2, 1)
    plt.pyplot.grid(True)
    plt.pyplot.hist([df_before.loc[(df_before['state']==1),"wage_st"],df_before.loc[(df_before['state']==0),'wage_st']], 
                    bins=np.arange(4.05, 5.85, 0.10),density=True, label=['NJ','PA'])
    plt.pyplot.xlabel('wage range')
    plt.pyplot.ylabel('percent of stores $*(10^{-1})$')
    plt.pyplot.legend(loc='upper right')
    plt.pyplot.title('February 1992')
    

    plt.pyplot.subplot(1, 2, 2)
    plt.pyplot.grid(True)
    plt.pyplot.hist([df_after.loc[(df_after['state']==1),'wage_st'],df_after.loc[(df_after['state']==0),'wage_st']], 
                bins=np.arange(4.05, 5.85, 0.10), density=True, label=['NJ','PA'])
    plt.pyplot.xlabel('wage range')
    plt.pyplot.ylabel('percent of stores $*(10^{-1})$')
    plt.pyplot.legend(loc='upper right')
    plt.pyplot.title('November 1992')

    return plt.pyplot.show()


def figures(df_before,df_after,sel_var):
    
    '''
    Function to generate distribution plot of selected variables
    
    argument:subset for pre and post intervention, variable(s) of interest
    return:distribution of the variable(s) of interest
    '''
    plt.pyplot.figure(figsize=(15, 10), dpi=70, facecolor='w', edgecolor='k')
    plt.pyplot.subplots_adjust(wspace=0.2, hspace=0.4)

    plt.pyplot.subplot(1, 2, 1)
    plt.pyplot.grid(True)
    plt.pyplot.hist([df_before.loc[(df_before['state']==1),sel_var],df_before.loc[(df_before['state']==0),sel_var]], label=['NJ','PA'])
    plt.pyplot.xlabel(sel_var)
    plt.pyplot.ylabel('count of stores')
    plt.pyplot.legend(loc='upper right')
    plt.pyplot.title('February 1992')
    

    plt.pyplot.subplot(1, 2, 2)
    plt.pyplot.grid(True)
    plt.pyplot.hist([df_after.loc[(df_after['state']==1),sel_var],df_after.loc[(df_after['state']==0),sel_var]], label=['NJ','PA'])
    plt.pyplot.xlabel(sel_var)
    plt.pyplot.ylabel('count of stores')
    plt.pyplot.legend(loc='upper right')
    plt.pyplot.title('November 1992')

    return plt.pyplot.show()