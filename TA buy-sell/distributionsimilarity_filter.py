""" Test the performance of the distribution similarity as a filter """
import matplotlib.pyplot as plt
from trading.processes import Simulation
from func import *
from ta_functions import Strategy
from copy import copy

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
        break

    return not all(v)


if __name__ == "__main__":
    s = Simulation(
        broker = "gbm",
        fiat = "mx",
        end = date(2021,12,31),
        simulations=36,
        parallel=True,
        commission=0, 
        verbose = 1
    )


    s.analyze(
        frequency="1m",
        test_time=1,
        analysis={
            "LR_DSFilterDown":{
                "function":lr_dsfilter,
                "time":72,
                "type":"filter"
            }
        },
        run = True
    )

    s.optimize(
        balance_time=24,
        exp_return="mean",
        risk = "1/N",
        objective = "minrisk",
        run = True
    )

    results = s.results_compilation()

    df = s.behaviour( results.loc[ 0, "route" ] )

    df[ "acc" ].plot()
    plt.show()