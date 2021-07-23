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



def lmplot(df,y):
    
    '''
    arguments:dataset,outcome variable
    return:regression plot condition on a third variable (i.e. restaurant chain) across different columns
    
    '''
    df['chains']=df['chain'].replace({1: 'Burger King', 2: 'KFC', 3: 'Roy Rogers', 4: 'Wendys'})#assign a name to each category
    
    fig = sns.lmplot(x='GAP', y = y, hue='chains',col='chains', col_wrap=2, aspect=1, x_jitter=.05,
                         data = df)
    x=''
    if 'change_in_log_price' in df.columns:
        x='change in log price'
    else:
        x='change in FTE'

    fig.set_axis_labels('GAP wage',x)
    
    pltpy.show(fig)
    
    
    
def get_shares_chain_stores():
    '''
    generate pie chart of chain stores distribution across state
    based on figures of 'table 2 - panel 1'
    '''
    labels = ["Burger King", "KFC", "Roy Rogers","Wendy's"]

    fig, (ax1, ax2) = pltpy.subplots(1, 2)
    ax1.pie([41.09, 20.54, 24.77,13.60], labels=labels, autopct="%1.2f%%")
    ax1.set_title("NJ")
    ax2.pie([44.30, 15.19, 21.52,18.99], labels=labels, autopct="%1.2f%%")
    ax2.set_title("PA")
    fig.tight_layout()
    

    
def get_common_support(df_control,df_treated):
    '''
    arguments:dataset for control group, dataset for treated group
    return:plot histohram of wage with kernel distribution
    '''
    bins=np.arange(4.05, 5.85, 0.10)
    fig=sns.histplot(data=df_treated,x='wage_st', bins=bins, label='treated', kde=True,color='blue',element="step")
    fig=sns.histplot(data=df_control,x='wage_st', bins=bins, label='control', kde=True,color='orange',element="step")
    fig.set_xlim([4, 5.75])
    fig.set_xlabel("wage range")
    fig.legend()    
    
    
    
    
def get_common_support2(df_control, df_treated):

    fig, ax = pltpy.subplots(1, 1)
    bins=np.arange(4.05, 5.85, 0.10)
    ax.hist([df_control.wage_st, df_treated.wage_st], bins=bins, label=["control", "treated"])
    ax.set_xlim([4, 5.75])
    ax.set_xlabel("wage range")
    ax.legend()
    
    
def scatter_corr(df,var1,var2):
    '''
    argument: dataset, x-axis variable, y-axis variable
    return:scatter plot with Pearson correlation coeff
    '''
    sns.jointplot(x=var1, y=var2, data=df)
    df_dropna=df[[var1,var2]].dropna()

    stat = scipy.stats.pearsonr(df_dropna[var1], df_dropna[var2])[0]#we are just interesting in the firs value from the output of pearsonr command
    print(f"The Pearson correlation coefficient is {stat:7.3f}")
    
    
    
    
def bar_chart_wage(df1,df2):
    
    low_pre=df1.loc[(df1['wage_st']==4.25),'FTE'].mean()
    high_pre=df1.loc[(df1['wage_st']>=5),'FTE'].mean()
    low_post=df2.loc[(df2['wage_st']==4.25) & (df2['state']==1),'FTE2'].mean()
    high_post=df2.loc[(df2['wage_st']>=5) & (df2['state']==1),'FTE2'].mean()

    data = [[low_pre, high_pre],
    [low_post, high_post]]
    X = np.arange(2)
    fig = pltpy.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    ax.bar(X + 0.25, data[1], color = 'r', width = 0.25)
    # Add some text for labels, title and custom x-axis tick labels, etc
    ax.set_ylabel('Average FTE Employment')
    ax.set_title('Average FTE Employment in Low- and High-wage areas of New Jersey, before and after 1992 minimum wage increase')
    ax.set_xticks(([0.12, 1.12]))
    ax.set_xticklabels(['Low Wage','High Wage'])
    ax.set_yticks(np.arange(4, 26, 4))
    ax.legend(labels=['Feb-Mar', 'Nov-Dec'], loc='best')
    fig.tight_layout()
    pltpy.show()
