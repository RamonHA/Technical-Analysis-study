""" Test the performance of the distribution similarity as a filter """
import matplotlib.pyplot as plt
from trading.processes import Simulation
from func import *
from ta_functions import Strategy
from copy import copy

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

import pandas as pd
import numpy as np
import collections

def dsfilter(asset):
    nasset = copy(asset)

    st = Strategy( asset=asset, tas = "trend_oneparam" )

    st.TREND_FIRST_PARAM = ( 3, 30, 3 )

    r = st.value( target=[ 1 ], verbose = 0 )

    r = r[ r["result"] < 0.05 ]

    if r.empty:
        return None
    
    r.sort_values( by = "result", ascending=True , inplace = True)

    l = 10 if len(r) > 10 else int(len(r) / 2)

    r = r.iloc[ :10 ]

    v = [ ]
    last_v = st.asset.df.iloc[-1].to_dict()

    for i, row in r.iterrows():
        v.append( row["range_down"][0] < last_v[ row["col"] ] < row["range_down"][1] )
    
    counter = collections.Counter(v)

    return (counter[False] > counter[True]) and (counter[False] > len( v ) // 2)

def common_tree(asset):
    st = Strategy( asset=asset, tas = "trend_oneparam" )

    # Fix parameters grid
    st.TREND_FIRST_PARAM = ( 3, 20, 3 )
    st.OSCILLATORS_FIRST_PARAM = (3, 20, 3)

    st.run_tas()

    df = st.asset.df
    df[ "target" ] = df[ "close" ].pct_change( 1 ).shift( -1 )

    df = df.replace( [np.inf, -np.inf], np.nan )
    
    names = df.isna().all()
    names = names[ names ].index.to_list()

    df.drop(columns = names, inplace = True)

    if len(df.dropna()) == 0:
        names = df.isna().sum()
        names = names[names > int(len(df) * 0.2)].index.to_list()
        df.drop(columns = names, inplace = True)

    df.dropna(inplace = True)

    train_size = int( len( df ) * 0.9 )
    train = df.iloc[ :train_size ]
    test = df.iloc[-train_size:]

    x_train = train.drop(columns = ["target"])
    y_train = train["target"]


def HalvingRandomForest(asset):

    

    halving_gs_results = HalvingGridSearchCV(
        rf,
        param_grid,
        cv=5,
        factor=3,
        min_resources='exhaust'
    ).fit(x_train, y_train)

    results = pd.DataFrame(halving_gs_results.cv_results_)
    results.loc[:, 'mean_test_score'] *= 100

    results = results.loc[:, ('iter', 'rank_test_score', 'mean_test_score', 'params')]
    results.sort_values(by=['iter', 'rank_test_score'], ascending=[False, True], inplace=True)

def BruteRandomForest(asset):



    rf = RandomForestRegressor()
    param_grid = {
        "n_estimators":[10, 20, 50, 100, 200, 500],
        "criterion":["poisson"],
        "bootstrap":[False],

    }

    

def lr_dsfilter(asset):

    nasset = copy(asset)

    st = Strategy( asset=asset, tas = "trend_oneparam" )

    st.TREND_FIRST_PARAM = ( 3, 30, 3 )

    r = st.value( target=[ 1 ], verbose = 0 )

    r = r[ r["result"] < 0.05 ]

    if r.empty:
        return None
    
    r.sort_values( by = "result", ascending=True , inplace = True)

    l = 10 if len(r) > 10 else int(len(r) / 2)

    r = r.iloc[ :10 ]

    v = [ ]
    last_v = st.asset.df.iloc[-1].to_dict()

    for i, row in r.iterrows():
        v.append( row["range_down"][0] < last_v[ row["col"] ] < row["range_down"][1] )
    
    counter = collections.Counter(v)

    if (counter[False] > counter[True]) and (counter[False] > len( v ) // 2):
        lr = LinearRegression()

        df = st.asset.df.copy()

        df[ "target" ] = df[ "close" ].pct_change( 1 ).shift( -1 )

        df = df.replace( [np.inf, -np.inf], np.nan )
        
        names = df.isna().all()
        names = names[ names ].index.to_list()

        df.drop(columns = names, inplace = True)

        to_predict = df.iloc[-1:]

        to_predict.fillna( 0, inplace = True )

        df.dropna(inplace = True)

        if len( df ) < 2:
            
            return None

        if to_predict.index[0] in df.index:
            df = df.iloc[:-1]

        lr.fit( df.drop(columns = ["target"]), df[["target"]] )

        p = lr.predict( to_predict.drop( columns = ["target"] ) )

        return p[-1][-1]

    return None


if __name__ == "__main__":
    s = Simulation(
        broker = "gbm",
        fiat = "mx",
        end = date(2021,12,31),
        simulations=36,
        parallel=False,
        commission=0, 
        verbose = 2
    )


    s.analyze(
        frequency="1m",
        test_time=1,
        analysis={
            "LR_DSFilterDown":{
                "function":lr_dsfilter,
                "time":72,
                "type":"prediction"
            }
        },
        run = False,
        cpus = 4
    )

    for r in [ "efficientfrontier" ]:#, "mad", "msv", "flpm", "slpm", "cvar", "evar", "wr", "mdd", "add", "cdar", "edar", "uci" ]:

        s.optimize(
            balance_time=26,
            exp_return=True,
            # value = 1,
            risk = r,
            objective = "efficientreturn",
            target_return = 0.01,
            run = False,
            filter = "highest",
            filter_qty = 10
        )


    results = s.results_compilation()

    df = s.behaviour( results.loc[ 0, "route" ] )

    df[ "acc" ].plot()
    plt.show()