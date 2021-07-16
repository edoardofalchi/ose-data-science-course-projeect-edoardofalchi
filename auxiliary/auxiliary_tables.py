"""This module contains auxiliary functions for the creation of tables in the main notebook."""

import json
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
from stargazer.stargazer import Stargazer
import math


from auxiliary.auxiliary_plots import *
from auxiliary.auxiliary_prepare_data import *


def did_est(variable,df):
    '''
    arguments:variable of interest, dataset
    return: 2x2 dif-in-dif table
    '''
    
    # NJ Before and after
    NJ_before = round(df.loc[(df['state']==1) & (df['time']!=1),variable].mean(),2)
    NJ_after  = round(df.loc[(df['state']==1) & (df['time']==1),variable].mean(),2)

    # PA Before and after
    PA_before = round(df.loc[(df['state']!=1) & (df['time']!=1),variable].mean(),2)
    PA_after  = round(df.loc[(df['state']!=1) & (df['time']==1),variable].mean(),2)
    
    did= pd.DataFrame(data=np.array([[NJ_after, NJ_before,NJ_after - NJ_before],
    [PA_after, PA_before, PA_after - PA_before]]), index=['NJ','PA'],columns=['after','before','$\Delta$'])
    
    return did


def table1(df):
    '''
    replication Table 1
    
    argument:dataset
    return: table1 from the paper
    '''
    state_unique= df["state"].unique()
    status_unique= df["status2"].unique()
    table1 = pd.DataFrame()
    for i in state_unique:
        for j in status_unique:
            table1.at[j,i]= sum((df.state == i) & (df.status2 == j))
    table1["All"]=table1[0]+table1[1]

    table1=table1.append(table1.sum(numeric_only=True), ignore_index=True)

    table1 = table1.rename({0: 'PA',1: 'NJ'}, axis=1)
    table1 = table1.rename({0: 'interviewed stores:',1: 'closed sores:',2:'temp closed -highway',
                        3:'under renovation:',4:'refusals:',5:'temp closed - fire',6:'total stores in sample:'}, axis=0)

    return table1



#table2 - panel 1
def distr_store_type(data_NJ,data_PA):
    '''
    replication Table 2 - panel 1
    
    argument:subset for New Jersey, subset for Pennsilvanya
    return:distribution of stores table2 from the paper
    '''
    variables=['bk','kfc','roys','wendys','co_owned']
    NJ= pd.DataFrame()
    PA= pd.DataFrame()
    #df_NJ= df[df["state"]==1]
    #df_PA= df[df["state"]==0]

    NJ['NJ']= (data_NJ[variables].sum() / len(data_NJ[variables])) * 100
    PA['PA']= (data_PA[variables].sum() / len(data_PA[variables])) * 100
    table2_1 = pd.concat([NJ, PA], axis=1)    
    table2_1['t-stat']=scipy.stats.ttest_ind(data_NJ[variables],data_PA[variables])[0]
    table2_1 = table2_1.astype(float).round(2)
    table2_1= table2_1.rename(index={'bk': 'Burger King','kfc':'KFC','roys':'Roy Rogers',
                                    'wendys':"Wendy's",'co_owned':'company-owned'})

    print("Distribution of store types (percentage)")
    return table2_1



#table2 - panel 2
def means_wave1(data_NJ,data_PA):
    '''
    replication Table 2 - panel 2
    
    argument:subset for New Jersey, subset for Pennsilvanya
    return:means in wave1 table2 from the paper
    '''
    variables=['FTE','pct_FTE','wage_st','price_full_meal','hoursopen']
    NJ= pd.DataFrame()
    PA= pd.DataFrame()
    #df_NJ_0= df[(df["state"]==1) & (df["time"]==0)]
    #df_PA_0= df[(df["state"]==0) & (df["time"]==0)]

    se_NJ=[]
    a=np.array(data_NJ['FTE'])
    a = a[~np.isnan(a)]
    b=np.array(data_NJ['pct_FTE'])
    b = b[~np.isnan(b)]
    c=np.array(data_NJ['wage_st'])
    c = c[~np.isnan(c)]
    d=np.array(data_NJ['price_full_meal'])
    d = d[~np.isnan(d)]
    e=np.array(data_NJ['hoursopen'])
    e = e[~np.isnan(e)]
    x=[a,b,c,d,e]
    for x in x:
        se_NJ.append(scipy.stats.sem(x))

    NJ['mean']= data_NJ[variables].mean()
    NJ['std_err']= se_NJ
    
    se_PA=[]
    f=np.array(data_PA['FTE'])
    f = f[~np.isnan(f)]
    g=np.array(data_PA['pct_FTE'])
    g = g[~np.isnan(g)]
    h=np.array(data_PA['wage_st'])
    h = h[~np.isnan(h)]
    l=np.array(data_PA['price_full_meal'])
    l = l[~np.isnan(l)]
    m=np.array(data_PA['hoursopen'])
    m = m[~np.isnan(m)]
    z=[f,g,h,l,m]
    for z in z:
        se_PA.append(scipy.stats.sem(z))    
    
    PA['mean']= data_PA[variables].mean()
    PA['std_err']= se_PA
    
   
    table2_2 = pd.concat([NJ, PA], axis=1)
    table2_2.columns = pd.MultiIndex.from_product([['stores in NJ', 'stores in PA'],
                                                ['mean', 'std_err']])
    table2_2['t-stat']=[scipy.stats.ttest_ind(a,f)[0],scipy.stats.ttest_ind(b,g)[0],
                   scipy.stats.ttest_ind(c,h)[0],scipy.stats.ttest_ind(d,l)[0],scipy.stats.ttest_ind(e,m)[0]]

    data_NJ['wage_4.25$']= data_NJ['wage_st']==4.25
    wage_425_NJ_0=(data_NJ['wage_4.25$'].sum() / len(data_NJ['wage_4.25$']))*100

    se_425_NJ_0= scipy.stats.sem(data_NJ['wage_4.25$'])*100

    data_PA['wage_4.25$']= data_PA['wage_st']==4.25
    wage_425_PA_0=(data_PA['wage_4.25$'].sum() / len(data_PA['wage_4.25$']))*100

    se_425_PA_0= scipy.stats.sem(data_PA['wage_4.25$'])*100

    test_wage_425 = scipy.stats.ttest_ind(data_NJ['wage_4.25$'],data_PA['wage_4.25$'])[0]

    table2_2.loc["wage_425",:]=(wage_425_NJ_0,se_425_NJ_0,wage_425_PA_0,se_425_PA_0,test_wage_425)#add as new row to dataframe

    table2_2 = table2_2.astype(float).round(2)
    table2_2= table2_2.rename(index={'FTE': 'FTE employment','pct_FTE':'% full-time employees','wage_st':'starting wage',
                                    'price_full_meal':'price of full meal','hoursopen':'hours open','wage_425':'wage = 4.25$ (%)'})


    print("Means in wave 1")
    return table2_2





#table2 - panel 3
def means_wave2(data_NJ,data_PA):
    '''
    replication Table 2 - panel 3
    
    argument:subset for New Jersey, subset for Pennsilvanya
    return:means in wave2 table2 from the paper
    '''
    variables=['FTE','pct_FTE','wage_st','price_full_meal','hoursopen']
    NJ= pd.DataFrame()
    PA= pd.DataFrame()
    #df_NJ_1= df[(df["state"]==1) & (df["time"]==1)]
    #df_PA_1= df[(df["state"]==0) & (df["time"]==1)]

    se_NJ=[]
    a=np.array(data_NJ['FTE'])
    a = a[~np.isnan(a)]
    b=np.array(data_NJ['pct_FTE'])
    b = b[~np.isnan(b)]
    c=np.array(data_NJ['wage_st'])
    c = c[~np.isnan(c)]
    d=np.array(data_NJ['price_full_meal'])
    d = d[~np.isnan(d)]
    e=np.array(data_NJ['hoursopen'])
    e = e[~np.isnan(e)]
    x=[a,b,c,d,e]
    for x in x:
        se_NJ.append(scipy.stats.sem(x))

    NJ['mean']= data_NJ[variables].mean()
    NJ['std_err']= se_NJ
    
    se_PA=[]
    f=np.array(data_PA['FTE'])
    f = f[~np.isnan(f)]
    g=np.array(data_PA['pct_FTE'])
    g = g[~np.isnan(g)]
    h=np.array(data_PA['wage_st'])
    h = h[~np.isnan(h)]
    l=np.array(data_PA['price_full_meal'])
    l = l[~np.isnan(l)]
    m=np.array(data_PA['hoursopen'])
    m = m[~np.isnan(m)]
    z=[f,g,h,l,m]
    for z in z:
        se_PA.append(scipy.stats.sem(z))    
    
    PA['mean']= data_PA[variables].mean()
    PA['std_err']= se_PA
    
   
    table2_3 = pd.concat([NJ, PA], axis=1)
    table2_3.columns = pd.MultiIndex.from_product([['stores in NJ', 'stores in PA'],
                                                ['mean', 'std_err']])
    table2_3['t-stat']=[scipy.stats.ttest_ind(a,f)[0],scipy.stats.ttest_ind(b,g)[0],
                   scipy.stats.ttest_ind(c,h)[0],scipy.stats.ttest_ind(d,l)[0],scipy.stats.ttest_ind(e,m)[0]]

    data_NJ['wage_4.25$']= data_NJ['wage_st']==4.25
    wage_425_NJ_1=(data_NJ['wage_4.25$'].sum() / len(data_NJ['wage_4.25$']))*100

    se_425_NJ_1= scipy.stats.sem(data_NJ['wage_4.25$'])*100

    data_PA['wage_4.25$']= data_PA['wage_st']==4.25
    wage_425_PA_1=(data_PA['wage_4.25$'].sum() / len(data_PA['wage_4.25$']))*100

    se_425_PA_1= scipy.stats.sem(data_PA['wage_4.25$'])*100

    
    table2_3.loc["wage_425",:]=(wage_425_NJ_1,se_425_NJ_1,wage_425_PA_1,se_425_PA_1,float("NaN"))#add as new row to dataframe
    
    
    data_NJ['wage_5.05$']= data_NJ['wage_st']==5.05
    wage_505_NJ_1=(data_NJ['wage_5.05$'].sum() / len(data_NJ['wage_5.05$']))*100

    se_505_NJ_1= scipy.stats.sem(data_NJ['wage_5.05$'])*100

    data_PA['wage_5.05$']= data_PA['wage_st']==5.05
    wage_505_PA_1=(data_PA['wage_5.05$'].sum() / len(data_PA['wage_5.05$']))*100

    se_505_PA_1= scipy.stats.sem(data_PA['wage_5.05$'])*100

    test_wage_505 = scipy.stats.ttest_ind(data_NJ['wage_5.05$'],data_PA['wage_5.05$'])[0]
    
    table2_3.loc["wage_505",:]=(wage_505_NJ_1,se_505_NJ_1,wage_505_PA_1,se_505_PA_1,test_wage_505)#add as new row to dataframe

    table2_3 = table2_3.astype(float).round(2)
    table2_3= table2_3.rename(index={'FTE': 'FTE employment','pct_FTE':'% full-time employees','wage_st':'starting wage',
                                    'price_full_meal':'price of full meal','hoursopen':'hours open','wage_425':'wage = 4.25$ (%)',
                                    'wage_505':'wage = 5.05$ (%)'})


    print("Means in wave 2")
    return table2_3




#### create table 3 from the paper###########
def get_table3(df_tab3,df_tab3_2,df_tab3_3,df_tab3_4,df_tab3_5):
    #####prepare data to fill in Table 3 according to the paper
    #row1 column 1-3
    stores_PA=df_tab3.loc[df_tab3['state']==0,'FTE']
    stores_NJ=df_tab3.loc[df_tab3['state']==1,'FTE']
    a=stores_PA.mean()
    b=stores_NJ.mean()
    c=b-a

    #row1 column 4-8
    NJ_low_wage= df_tab3_2.loc[(df_tab3_2['wage_st']==4.25) & (df_tab3_2['state']==1),'FTE']
    NJ_med_wage= df_tab3_2.loc[(df_tab3_2['wage_st']>=4.26) & (df_tab3_2['wage_st']<=4.99) & (df_tab3_2['state']==1),'FTE']
    NJ_high_wage= df_tab3_2.loc[(df_tab3_2['wage_st']>=5) & (df_tab3_2['state']==1),'FTE']
    d=NJ_low_wage.mean()
    e=NJ_med_wage.mean()
    f=NJ_high_wage.mean()
    g=d-f
    h=e-f
    #a,b,c,d,e,f,g,h

    #row2 column 1-3
    stores_PA2=df_tab3_3.loc[df_tab3_3['state']==0,'FTE2']
    stores_NJ2=df_tab3_3.loc[df_tab3_3['state']==1,'FTE2']
    a2=stores_PA2.mean()
    b2=stores_NJ2.mean()
    c2=b2-a2

    #row2 column 4-8
    NJ_low_wage2= df_tab3_2.loc[(df_tab3_2['wage_st']==4.25) & (df_tab3_2['state']==1),'FTE2']
    NJ_med_wage2= df_tab3_2.loc[(df_tab3_2['wage_st']>=4.26) & (df_tab3_2['wage_st']<=4.99) & (df_tab3_2['state']==1),'FTE2']
    NJ_high_wage2= df_tab3_2.loc[(df_tab3_2['wage_st']>=5) & (df_tab3_2['state']==1),'FTE2']

    d2=NJ_low_wage2.mean()
    e2=NJ_med_wage2.mean()
    f2=NJ_high_wage2.mean()
    g2=d2-f2
    h2=e2-f2
    #a2,b2,c2,d2,e2,f2,g2,h2

    #row3 column 1-8
    a3=a2-a
    b3=b2-b
    c3=c2-c
    d3=d2-d
    e3=e2-e
    f3=f2-f
    g3=g2-g
    h3=h2-h
    #a3,b3,c3,d3,e3,f3,g3,h3

    #row4 column 1-3
    stores_PA=df_tab3_4.loc[df_tab3_4['state']==0,'FTE']
    stores_NJ=df_tab3_4.loc[df_tab3_4['state']==1,'FTE']
    a_=stores_PA.mean()
    b_=stores_NJ.mean()
    c_=b_-a_

    stores_PA2=df_tab3_4.loc[df_tab3_4['state']==0,'FTE2']
    stores_NJ2=df_tab3_4.loc[df_tab3_4['state']==1,'FTE2']
    a2_=stores_PA2.mean()
    b2_=stores_NJ2.mean()
    c2_=b2_-a2_

    a4=a2_-a_
    b4=b2_-b_
    c4=c2_-c_

    #row4 column 4-8
    NJ_low_wage= df_tab3_5.loc[(df_tab3_5['wage_st']==4.25) & (df_tab3_5['state']==1),'FTE']
    NJ_med_wage= df_tab3_5.loc[(df_tab3_5['wage_st']>=4.26) & (df_tab3_5['wage_st']<=4.99) & (df_tab3_5['state']==1),'FTE']
    NJ_high_wage= df_tab3_5.loc[(df_tab3_5['wage_st']>=5) & (df_tab3_5['state']==1),'FTE']
    d_=NJ_low_wage.mean()
    e_=NJ_med_wage.mean()
    f_=NJ_high_wage.mean()
    g_=d_-f_
    h_=e_-f_

    NJ_low_wage2= df_tab3_5.loc[(df_tab3_5['wage_st']==4.25) & (df_tab3_5['state']==1),'FTE2']
    NJ_med_wage2= df_tab3_5.loc[(df_tab3_5['wage_st']>=4.26) & (df_tab3_5['wage_st']<=4.99) & (df_tab3_5['state']==1),'FTE2']
    NJ_high_wage2= df_tab3_5.loc[(df_tab3_5['wage_st']>=5) & (df_tab3_5['state']==1),'FTE2']
    d2_=NJ_low_wage2.mean()
    e2_=NJ_med_wage2.mean()
    f2_=NJ_high_wage2.mean()
    g2_=d2_-f2_
    h2_=e2_-f2_

    d4=d2_-d_
    e4=e2_-e_
    f4=f2_-f_
    g4=g2_-g_
    h4=h2_-h_
    #a4,b4,c4,d4,e4,f4,g4,h4
    row1=[a,b,c,d,e,f,g,h]
    row2=[a2,b2,c2,d2,e2,f2,g2,h2]
    row3=[a3,b3,c3,d3,e3,f3,g3,h3]
    row4=[a4,b4,c4,d4,e4,f4,g4,h4]
    
    table3 = pd.DataFrame(
        {
            "PA ($i$)": [],
            "NJ ($ii$)": [],
            "difference, NJ - PA ($iii$)": [],
            "wage=\$4.25 ($iv$)": [],
            "wage=\$4.26-\$4.99 ($v$)": [],
            "wage>=\$5 ($vi$)": [],
            "low - high ($vii$)":[],
            "midrange - high ($viii$)":[]
        }
    )

    table3["variables"] = ["FTE employment before, all available observations",
                      "FTE employment after, all available observations",
                      "change in mean FTE employment",
                      "change in mean FTE employment, balanced sample of stores"]
    table3 = table3.set_index("variables")
    table3.iloc[0]=row1
    table3.iloc[1]=row2
    table3.iloc[2]=row3
    table3.iloc[3]=row4
    table3=table3.round(2)
    tuples=[('Stores by state     ',"PA"),('Stores by state     ',"NJ"),('Stores by state     ',"difference, NJ - PA"),('     Stores in New Jersey',"wage=\$4.25"),
        ('     Stores in New Jersey',"wage=\$4.26-\$4.99"),('     Stores in New Jersey',"wage>=\$5"),('Differences within NJ',"low - high"),
        ('Differences within NJ',"midrange - high")]
    table3.columns = pd.MultiIndex.from_tuples(tuples)

    print('Average employment per store before and after the rise in New Jersey minimum wage')
    return table3



################create the table generation function##########################################

def get_table_4and7(dependent_var, data):
    '''
    argument:dependent variable, dataset
    return:either table4 or table7 depending on the input dataset
    '''
    model_1=sm_api.OLS(data[dependent_var], sm_api.add_constant(data["state"])).fit()
    model_2=sm_api.OLS(data[dependent_var], sm_api.add_constant(data[["state","bk","kfc","roys","co_owned"]])).fit()
    model_3=sm_api.OLS(data[dependent_var], sm_api.add_constant(data["GAP"])).fit()
    model_4=sm_api.OLS(data[dependent_var], sm_api.add_constant(data[["GAP","bk","kfc","roys","co_owned"]])).fit()
    model_5=sm_api.OLS(data[dependent_var], sm_api.add_constant(data[["GAP","bk","kfc","roys","co_owned","southj","centralj","pa1","pa2"]])).fit()
    Table=Stargazer([model_1,model_2,model_3,model_4,model_5])
    Table.rename_covariates({'state': 'New Jersey dummy', 'GAP':'Initial wage GAP' })
    Table.add_line('Controls for chain and ownership', ['No', 'Yes','No','Yes','Yes'])
    Table.add_line('Controls for region', ['No', 'No','No','No','Yes'])
    F2= model_2.f_test('(state = 0), (bk = 0), (kfc = 0), (roys =0),(co_owned= 0),(const=0)').pvalue.round(3)
    F4= model_4.f_test('(GAP = 0), (bk = 0), (kfc = 0), (roys =0),(co_owned= 0),(const=0)').pvalue.round(3)
    F5= model_5.f_test('(GAP = 0), (bk = 0), (kfc = 0), (roys =0),(co_owned= 0),(const=0), (southj=0),(centralj=0),(pa1=0),(pa2=0)').pvalue.round(3)
    if dependent_var=="change_in_FTE":
        Table.add_line('Probability value for controls', ['-', F2,'-',F4,F5])
    Table.title("Models for " + dependent_var )
    Table.covariate_order(['state', 'GAP'])
    print("The mean and standard deviation of the dependent variable are", data[dependent_var].mean(),"and", data[dependent_var].std(),",respectively.")

    return Table







###function generation for table 5
def get_table5(df_tab5,df_extended):
    
    
    df_tab5['prop_change_FTE']=df_tab5['change_in_FTE']/((df_tab5['FTE']+df_tab5['FTE2'])/2) #according to the paper, the proportional change in employment is defined as the change in employment divided by the average level of employment in waves 1 and 2.

    outcome=['change_in_FTE','prop_change_FTE']
    covariate=['state','GAP']

    nj_coef1=[]
    nj_se1=[]
    gap_coef2=[]
    gap_se2=[]
    nj_coef3=[]
    nj_se3=[]
    gap_coef4=[]
    gap_se4=[]


                                    ###fill in the table###
    ###row1 base specification###
    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])

    ####row2 Treat four temporarily closed stores as permanently closed###
    df_tab5_row2=df_extended[(~df_extended.FTE.isna()) & (~df_extended.wage_st.isna()) & (~df_extended.FTE2.isna())
                       & (~df_extended.wage_st2.isna()) | (df_extended.status2 >=2)]
      #Wave-2 employment at four temporarily closed stores is set to 0 (rather than missing).
    df_tab5_row2.loc[df_tab5_row2['status2'] == 2, 'FTE2'] = 0
    df_tab5_row2.loc[df_tab5_row2['status2'] == 4, 'FTE2'] = 0
    df_tab5_row2.loc[df_tab5_row2['status2'] == 5, 'FTE2'] = 0

    df_tab5_row2['change_in_FTE']=(df_tab5_row2["FTE2"] - df_tab5_row2["FTE"])
    df_tab5_row2['prop_change_FTE']=df_tab5_row2['change_in_FTE']/((df_tab5_row2['FTE']+df_tab5_row2['FTE2'])/2)

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row2).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row2).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row2).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row2).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row3 exclude managers in employment count###
    df_tab5_row3=df_tab5.copy()
      #full-time equivalent employment excludes managers and assistant managers.
    df_tab5_row3["FTE"] = df_tab5_row3.empft + 0.5 * df_tab5_row3.emppt
    df_tab5_row3["FTE2"] = df_tab5_row3.empft2 + 0.5 * df_tab5_row3.emppt2

    df_tab5_row3['change_in_FTE']=(df_tab5_row3["FTE2"] - df_tab5_row3["FTE"])
    df_tab5_row3['prop_change_FTE']=df_tab5_row3['change_in_FTE']/((df_tab5_row3['FTE']+df_tab5_row3['FTE2'])/2)

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row3).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row3).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row3).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row3).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row4 weight part-time as 0.4*full-time###
    df_tab5_row4=df_tab5.copy()
      #Full-time equivalent employment equals number of managers, assistant managers, and full-time nonmanagement workers, plus 0.4 times the number of part-time nonmanagement workers.
    df_tab5_row4["FTE"] = df_tab5_row4.empft + df_tab5_row4.nmgrs + 0.4 * df_tab5_row4.emppt
    df_tab5_row4["FTE2"] = df_tab5_row4.empft2 + df_tab5_row4.nmgrs + 0.4 * df_tab5_row4.emppt2

    df_tab5_row4['change_in_FTE']=(df_tab5_row4["FTE2"] - df_tab5_row4["FTE"])
    df_tab5_row4['prop_change_FTE']=df_tab5_row4['change_in_FTE']/((df_tab5_row4['FTE']+df_tab5_row4['FTE2'])/2)

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row4).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row4).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row4).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row4).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row5 weight part-time as 0.6*full-time###
    df_tab5_row5=df_tab5.copy()
      #Full-time equivalent employment equals number of managers, assistant managers, and full-time nonmanagement workers, plus 0.6 times the number of part-time nonmanagement workers.
    df_tab5_row5["FTE"] = df_tab5_row5.empft + df_tab5_row5.nmgrs + 0.6 * df_tab5_row5.emppt
    df_tab5_row5["FTE2"] = df_tab5_row5.empft2 + df_tab5_row5.nmgrs + 0.6 * df_tab5_row5.emppt2

    df_tab5_row5['change_in_FTE']=(df_tab5_row5["FTE2"] - df_tab5_row5["FTE"])
    df_tab5_row5['prop_change_FTE']=df_tab5_row5['change_in_FTE']/((df_tab5_row5['FTE']+df_tab5_row5['FTE2'])/2)

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row5).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row5).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row5).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row5).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])

    ###row6 exclude stores in NJ shore area###
    df_tab5_row6=df_extended.copy()
    #Sample excludes 35 stores located in towns along the New Jersey shore.
    df_tab5_row6=df_tab5_row6[(df_tab5_row6.shore==0)]
    df_tab5_row6=df_tab5_row6[(~df_tab5_row6.FTE.isna()) & (~df_tab5_row6.wage_st.isna()) & (~df_tab5_row6.FTE2.isna())
                       & (~df_tab5_row6.wage_st2.isna()) | (df_tab5_row6.status2 == 3)]

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row6).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row6).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row6).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row6).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])

    ###row7 add controls for wave-2 interview date###
    df_tab5_row7=df_tab5.copy()
    #models include three dummy variables identifying week of wave-2 interview in November-December 1992.
    #the authors do not specify how these dummies are constructed, so I arbitrarily split the time window in 3 intervals
    #containing 182, 144, 30 observations respectively while trying to keep a uniform bandwidth at the same time.
    df_tab5_row7['week1']=df_tab5_row7['date2'].between('1992-11-01','1992-11-13')
    df_tab5_row7['week2']=df_tab5_row7['date2'].between('1992-11-14','1992-12-08')
    df_tab5_row7['week3']=df_tab5_row7['date2'].between('1992-12-09','1992-12-31')

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned + week1 + week2", data=df_tab5_row7).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned +week1 + week2", data=df_tab5_row7).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned + week1 + week2", data=df_tab5_row7).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned + week1 + week2", data=df_tab5_row7).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row 8 exclude stores called more than twice in wave 1###
    df_tab5_row8=df_extended.copy()
    #Sample excludes 70 stores (69 in New Jersey) that were contacted three or more times before obtaining the wave-1 interview.
    df_tab5_row8=df_tab5_row8[(df_tab5_row8.ncalls<3)]
    df_tab5_row8=df_tab5_row8[(~df_tab5_row8.FTE.isna()) & (~df_tab5_row8.wage_st.isna()) & (~df_tab5_row8.FTE2.isna())
                       & (~df_tab5_row8.wage_st2.isna()) | (df_tab5_row8.status2 == 3)]

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row8).fit()
    nj_coef1.append(fit.params[1])
    nj_se1.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[0]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row8).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ", data=df_tab5_row8).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ", data=df_tab5_row8).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row 9 weight by initial employment###
    df_tab5_row9=df_tab5.copy()
    #Regression models are estimated by Weighted Least Squares, using employment in wave 1 as a weight.

    gap_coef2.append(np.nan)
    gap_se2.append(np.nan)

    fit=smf.wls(formula=f"{outcome[1]} ~ {covariate[0]} + bk + kfc + roys + co_owned ",weights=df_tab5_row9['FTE'], data=df_tab5_row9).fit()
    nj_coef3.append(fit.params[1])
    nj_se3.append(fit.bse[1])

    fit=smf.wls(formula=f"{outcome[1]} ~ {covariate[1]} + bk + kfc + roys + co_owned ",weights=df_tab5_row9['FTE'], data=df_tab5_row9).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])


    ###row 10 Stores in towns around Newark. I don't have zip codes in order to replicate findings###
    ###row 11 Stores in towns around Camden. I don't have zip codes in order to replicate findings###


    ###row 12 Subsample of Pennsylvania stores only###

    df_tab5_row12=df_extended.copy()

    df_tab5_row12=df_tab5_row12[(df_tab5_row12.state==0)]
    df_tab5_row12=df_tab5_row12[(~df_tab5_row12.FTE.isna()) & (~df_tab5_row12.wage_st.isna()) & (~df_tab5_row12.FTE2.isna())
                       & (~df_tab5_row12.wage_st2.isna()) | (df_tab5_row12.status2 == 3)]
    #incorrectly misspecify, on purpose, the GAP variable for Pennsylvania stores as the proportional increase in wages necessary to raise the wage to $5.05 per hour
    df_tab5_row12['GAP_adj']=np.where(df_tab5_row12['wage_st']>=5.05,0,((5.05 - df_tab5_row12['wage_st'])/ df_tab5_row12['wage_st']))




    fit=smf.ols(formula=f"{outcome[0]} ~ GAP_adj + bk + kfc + roys + co_owned ", data=df_tab5_row12).fit()
    gap_coef2.append(fit.params[1])
    gap_se2.append(fit.bse[1])

    fit=smf.ols(formula=f"{outcome[1]} ~ GAP_adj + bk + kfc + roys + co_owned ", data=df_tab5_row12).fit()
    gap_coef4.append(fit.params[1])
    gap_se4.append(fit.bse[1])

    nj_coef1.append(np.nan)
    nj_coef1.append(np.nan)

    nj_se1.append(np.nan)
    nj_se1.append(np.nan)


    nj_coef3.append(np.nan)

    nj_se3.append(np.nan)
    
    table5 = pd.DataFrame(
                {
                    "NJ dummy coeff.": nj_coef1,
                    "NJ dummy std.err.": nj_se1,
                    "Gap measure coeff.": gap_coef2,
                    "Gap measure std.err.": gap_se2,
                    "NJ dummy coeff": nj_coef3,
                    "NJ dummy std.err": nj_se3,
                    "Gap measure coeff":gap_coef4,
                    "Gap measure std.err":gap_se4
                }
            )

    table5["specification"] = ['base specification','treat 4 temporarily closed stores as permanently closed',
                                      'exclude managers in employment count','weight part-time as 0.4*full-time',
                                      'weight part-time as 0.6*full-time','exclude stores in NJ shore area','add controls for wave2 interview date',
                                      'exclude stores called more than twice in wave1','WLS: weight by initial employment','Pennsylvania stores only']
    table5 = table5.set_index("specification")

    table5=table5.round(2)
    table5= table5.replace(np.nan,'-')
    table5.columns = pd.MultiIndex.from_product([['Change in employment', 'Proportional change in employment'],['NJ dummy coeff.', 'NJ dummy std.err.','Gap measure coeff.','Gap measure std.err.']])


    print('Specification Tests of reduced-form employment models')
    return table5








def get_table6_panel1(df_tab6):
    ####create the variables needed for table 6
    df_tab6['frac_full_time_workers']=(df_tab6.empft / df_tab6.FTE)* 100 #fraction of part-time employees in total full-time-equivalent employment.
    df_tab6['frac_full_time_workers2']=(df_tab6.empft2 / df_tab6.FTE2)* 100
    #df_tab6['GAP']=np.where(df_tab6['wage_st']>=5.05,0,((5.05 - df_tab6['wage_st'])/ df_tab6['wage_st'])) #create GAP variable used in the paper page 779
    #df_tab6.loc[df_tab6['state'] == 0, 'GAP'] = 0 #GAP variable is set to 0 for Pennsilvanya state
    df_tab6["meals"] = df_tab6["meals"].astype("category")
    df_tab6["meals2"] = df_tab6["meals2"].astype("category")
    df_tab6=pd.concat([df_tab6.iloc[:,1], pd.get_dummies(df_tab6, columns=['meals','meals2'])],axis=1)
    df_tab6=df_tab6.rename(columns={'meals_1': 'free_meal','meals_2':'reduced_price_meal','meals_3': 'both_free_and_reduced','meals2_0.0':'no_meal2','meals2_1.0':'free_meal2',
    'meals2_2.0':'reduced_price_meal2','meals2_3.0':'both_free_and_reduced2'})
    df_tab6['slope_wage']=(df_tab6.firstinc / df_tab6.inctime)*100#slope of the wage profile, which is measured by the ratio of the typical first raise to the amount of time until the first raise is given.
    df_tab6['slope_wage2']=(df_tab6.firstin2 / df_tab6.inctime2)*100
    df_tab6= df_tab6.rename(columns={'firstin2': 'firstinc2'})


    lista=['hrsopen','nregs','nregs11','inctime','firstinc','reduced_price_meal','free_meal','both_free_and_reduced']

    outcome_measure=[]
    for i in lista:
        df_tab6[f'change_in_{i}']=df_tab6[f'{i}2']-df_tab6[i]
        outcome_measure.append(f'change_in_{i}')
    outcome_measure

    #column (i) Store characteristics and Wage profile
    nj_col=[]
    se_nj=[]
    for i in outcome_measure[0:5]:
        x=(df_tab6.loc[df_tab6['state']==1,i])
        nj_col.append(x.mean())

        z=np.array(x)
        z=z[~np.isnan(z)]
        se_nj.append(scipy.stats.sem(z))

    #column (ii) Store characteristics and Wage profile   
    pa_col=[]
    se_pa=[]
    for i in outcome_measure[0:5]:
        x=(df_tab6.loc[df_tab6['state']==0,i])
        pa_col.append(x.mean())

        z=np.array(x)
        z=z[~np.isnan(z)]
        se_pa.append(scipy.stats.sem(z))

    #column (iii) Store characteristics and Wage profile  
    zipped_lists_mean = zip(nj_col, pa_col) 
    diff_col=[x - y for (x, y) in zipped_lists_mean]

    zipped_lists_se = zip(se_nj, se_pa) 
    diff_col_se=[math.sqrt(x**2 + y**2) for (x, y) in zipped_lists_se]


    #column (i) Employee meal program
    nj_colII=[]
    se_njII=[]
    for i in lista[5:]:
        x=((df_tab6.loc[df_tab6['state']==1,f'{i}2']).sum() / len((df_tab6.loc[df_tab6['state']==1,f'{i}2']))*100) - ((df_tab6.loc[df_tab6['state']==1,i]).sum() / len((df_tab6.loc[df_tab6['state']==1,i]))*100)
        nj_colII.append(x.mean())

    for _ in outcome_measure[5:]:
        z=np.array(df_tab6.loc[df_tab6['state']==1,_])
        z=z[~np.isnan(z)]
        se_njII.append(np.sqrt(scipy.stats.sem(z)))

    #column (ii) Employee meal program
    pa_colII=[]
    se_paII=[]
    for i in lista[5:]:
        x=((df_tab6.loc[df_tab6['state']==0,f'{i}2']).sum() / len((df_tab6.loc[df_tab6['state']==0,f'{i}2']))*100) - ((df_tab6.loc[df_tab6['state']==0,i]).sum() / len((df_tab6.loc[df_tab6['state']==0,i]))*100)
        pa_colII.append(x.mean())

    for _ in outcome_measure[5:]:
        z=np.array(df_tab6.loc[df_tab6['state']==0,_])
        z=z[~np.isnan(z)]
        se_paII.append(np.sqrt(scipy.stats.sem(z)))

    #column (iii) Employee meal program
    zipped_lists_meanII = zip(nj_colII, pa_colII) 
    diff_colII=[x - y for (x, y) in zipped_lists_meanII]

    zipped_lists_seII = zip(se_njII, se_paII) 
    diff_col_seII=[math.sqrt(x**2 + y**2) for (x, y) in zipped_lists_seII]

    for _ in nj_colII:
        nj_col.append(_)
    for _ in se_njII:
        se_nj.append(_)
    for _ in pa_colII:
        pa_col.append(_)
    for _ in se_paII:
        se_pa.append(_)
    for _ in diff_colII:
        diff_col.append(_)
    for _ in diff_col_seII:
        diff_col_se.append(_)
    
    table6_1 = pd.DataFrame(
        {
            "NJ mean change": nj_col,
            "NJ std.err.": se_nj,
            "PA mean change": pa_col,
            "PA std.err.": se_pa,
            "NJ-PA mean change": diff_col,
            "NJ-PA std.err.": diff_col_se
            
        }
    )

    table6_1["Outcome measure"] = outcome_measure
    table6_1 = table6_1.set_index("Outcome measure")
    table6_1= table6_1.rename(index={'change_in_hrsopen': 'number of hours open per weekday','change_in_nregs':'number of cash registers','change_in_nregs11':'number of cash registers open at 11 a.m.',
                                    'change_in_inctime':'time to first raise (weeks)','change_in_firstinc':'usual amount of first raise (cents)','change_in_reduced_price_meal':'low price meal program',
                                    'change_in_free_meal':'free meal program','change_in_both_free_and_reduced':'combination of low price and free meals'})
    table6_1=table6_1.round(2)
 
    print('mean change in outcome')
    return table6_1



def get_table6_panel2(df_tab6):
    
     ####create the variables needed for table 6
    df_tab6['frac_full_time_workers']=(df_tab6.empft / df_tab6.FTE)* 100 #fraction of part-time employees in total full-time-equivalent employment.
    df_tab6['frac_full_time_workers2']=(df_tab6.empft2 / df_tab6.FTE2)* 100
    #df_tab6['GAP']=np.where(df_tab6['wage_st']>=5.05,0,((5.05 - df_tab6['wage_st'])/ df_tab6['wage_st'])) #create GAP variable used in the paper page 779
    #df_tab6.loc[df_tab6['state'] == 0, 'GAP'] = 0 #GAP variable is set to 0 for Pennsilvanya state
    df_tab6["meals"] = df_tab6["meals"].astype("category")
    df_tab6["meals2"] = df_tab6["meals2"].astype("category")
    df_tab6=pd.concat([df_tab6.iloc[:,1], pd.get_dummies(df_tab6, columns=['meals','meals2'])],axis=1)
    df_tab6=df_tab6.rename(columns={'meals_1': 'free_meal','meals_2':'reduced_price_meal','meals_3': 'both_free_and_reduced','meals2_0.0':'no_meal2','meals2_1.0':'free_meal2',
    'meals2_2.0':'reduced_price_meal2','meals2_3.0':'both_free_and_reduced2'})
    df_tab6['slope_wage']=(df_tab6.firstinc / df_tab6.inctime)*100#slope of the wage profile, which is measured by the ratio of the typical first raise to the amount of time until the first raise is given.
    df_tab6['slope_wage2']=(df_tab6.firstin2 / df_tab6.inctime2)*100
    df_tab6= df_tab6.rename(columns={'firstin2': 'firstinc2'})


    lista=['hrsopen','nregs','nregs11','inctime','firstinc','reduced_price_meal','free_meal','both_free_and_reduced']

    outcome_measure=[]
    for i in lista:
        df_tab6[f'change_in_{i}']=df_tab6[f'{i}2']-df_tab6[i]
        outcome_measure.append(f'change_in_{i}')
    outcome_measure

        #column (iv)
    nj_dummy_par=[]
    nj_dummy_se=[]
    for i in outcome_measure[0:5]:
        fit=smf.ols(formula=f"{i} ~ state + bk + kfc + roys + co_owned", data=df_tab6).fit()
        nj_dummy_par.append(fit.params[1])
        nj_dummy_se.append(fit.bse[1])

    #column (v)
    gap_a_par=[]
    gap_a_se=[]
    for i in outcome_measure[0:5]:
        fit=smf.ols(formula=f"{i} ~ GAP + bk + kfc + roys + co_owned", data=df_tab6).fit()
        gap_a_par.append(fit.params[1])
        gap_a_se.append(fit.bse[1])

    #column (vi)
    gap_b_par=[]
    gap_b_se=[]
    for i in outcome_measure[0:5]:
        fit=smf.ols(formula=f"{i} ~ GAP + bk + kfc + roys + co_owned + southj + centralj + pa1 + pa2", data=df_tab6).fit()
        gap_b_par.append(fit.params[1])
        gap_b_se.append(fit.bse[1])
    
    table6_2 = pd.DataFrame(
        {
            "NJ dummy coeff": nj_dummy_par,
            "NJ dummy std.err.": nj_dummy_se,
            "wage GAP coeff": gap_a_par,
            "wage GAP std.err.": gap_a_se,
            "wage GAP coeff (more control dummies) ": gap_b_par,
            "wage GAP std.err. (more control dummies)": gap_b_se
            
        }
    )

    table6_2["Outcome measure"] = outcome_measure[0:5]
    table6_2 = table6_2.set_index("Outcome measure")
    table6_2= table6_2.rename(index={'change_in_hrsopen': 'number of hours open per weekday','change_in_nregs':'number of cash registers','change_in_nregs11':'number of cash registers open at 11 a.m.',
                                    'change_in_inctime':'time to first raise (weeks)','change_in_firstinc':'usual amount of first raise (cents)'})
    table6_2=table6_2.round(2)
 
    print('Regression of change in outcome variable on:')
    return  table6_2
