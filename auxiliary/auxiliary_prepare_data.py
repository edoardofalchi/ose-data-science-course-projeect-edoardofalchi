"""This module contains auxiliary functions for calling the dataset in the main notebook."""

import pandas as pd
import numpy as np


from auxiliary.auxiliary_tables import *
from auxiliary.auxiliary_plots import *


def prepare_data_long():
    '''
    long-format  Card & Krueger (1994) dataset
    '''
    df = pd.read_stata('data/CK1994.dta')
    df = df.sort_values(by="store").set_index("store")
    df["chain"] = df["chain"].astype("category")
    df["meals"] = df["meals"].astype("category")
   # df.iloc[:,1:9] = df.iloc[:,1:9].astype('int')
   # df["time"] = df["time"].astype('int')
    df["FTE"] = df.empft + df.nmgrs + 0.5 * df.emppt
    df["pct_FTE"] = (df.empft / df.FTE)* 100
    df["nj_after"] = df.state * df.time #generate interaction dummy for New Jersey after min wage increase 
    df["price_full_meal"] = df.pricesoda + df.pricefry + df.priceentree
    df=pd.concat([df.iloc[:,0], pd.get_dummies(df, columns=['chain'])],axis=1)#create dummies for each fast-food chain
    df=df.rename(columns={'chain_1.0': 'bk','chain_2.0':'kfc','chain_3.0': 'roys','chain_4.0':'wendys'})
    return df


def prepare_data_wide():
    
    '''
    wide-format  Card & Krueger (1994) dataset
    '''
    df_extended = pd.read_stata('data/fastfood.dta')

    df_extended["date2"] = pd.to_datetime(df_extended["date2"].astype(str), format="%m%d%y")
    df_extended["FTE"] = df_extended.empft + df_extended.nmgrs + 0.5 * df_extended.emppt
    df_extended["FTE2"] = df_extended.empft2 + df_extended.nmgrs2 + 0.5 * df_extended.emppt2
    df_extended["chain"] = df_extended["chain"].astype("category")
    df_extended=pd.concat([df_extended.iloc[:,1], pd.get_dummies(df_extended, columns=['chain'])],axis=1)
    df_extended=df_extended.rename(columns={'chain_1': 'bk','chain_2':'kfc','chain_3': 'roys','chain_4':'wendys'})
    df_extended['GAP']=np.where(df_extended['wage_st']>=5.05,0,((5.05 - df_extended['wage_st'])/ df_extended['wage_st'])) #create GAP variable used in the paper page 779
    df_extended.loc[df_extended['state'] == 0, 'GAP'] = 0 #GAP variable is set to 0 for Pennsilvanya state
    df_extended['change_in_FTE']=(df_extended["FTE2"] - df_extended["FTE"])
    df_extended['prop_change_FTE']=df_extended['change_in_FTE']/((df_extended['FTE']+df_extended['FTE2'])/2)
    return df_extended