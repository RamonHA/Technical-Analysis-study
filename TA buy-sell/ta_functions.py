from trading import Asset
from datetime import date
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

TREND_FIRST_PARAM = ( 5, 200, 5 )
TREND_ONE_PARAM_NAMES = [ "sma", "ema", "wma" , "dema"]

def trend(asset):
    """ Trend followers """

    TREND_ONE_PARAM = {
        "sma":asset.sma,
        "ema":asset.ema,
        "wma":asset.wma,
        "dema":asset.dema
    }

    for name, operation in TREND_ONE_PARAM.items():
        
        for i in range( *TREND_FIRST_PARAM ):
            asset.df[ f"{name}_{i}" ] = operation( i ).pct_change(3)
    
    return asset

def prob(asset, col, target = 1, **kwargs):

    s = asset.df[ asset.df["target"] == target ][ col ].dropna()

    if len(s) ==0:
        print("No data")
        return 

    kde = stats.gaussian_kde(  s)

    xmin, xmax = kwargs.get("limits", (0.0, 0.15))
    
    integral, err = integrate.quad(kde, xmin, xmax)
    if kwargs.get("plot", False):
        x = np.linspace( kwargs.get("plot_min", -0.1) , kwargs.get("plot_max", xmax + 0.05)  ,100)
        
        x_integral = np.linspace(xmin, xmax, 100)

        plt.plot(x, kde(x), label="KDE")
        plt.fill_between(x_integral, 0, kde(x_integral),
                        alpha=0.3, color='b', label="Area: {:.3f}".format(integral))
        plt.legend()
    
    return integral



def get_prob( nasset, **kwargs ):
    data = []
    for i in range( *TREND_FIRST_PARAM ):
        c = "{}_{}".format(  kwargs.get("ta", "sma"), i )
        data.append( [i, len(nasset.df[c].dropna()) , prob( nasset, col = c , **kwargs) ] )

    data = pd.DataFrame( data , columns = ["length", "largo", "area"])
    data.set_index("length", inplace = True)

    return data

def get_plot(nasset, **kwargs):
    """ Plot the probabilities chart """
    data = get_prob( nasset, **kwargs )
    print( data["area"].idxmax(), data["area"].max())
    data["area"].plot(title = kwargs.get("ta", "sma"))

def new():
    """ Get asset """
    asset = Asset(
        symbol = "AAPL",
        broker="gbm",
        fiat = "mx",
        frequency="1d",
        start = date(2019,1,1),
        end = date(2022,5,1),
        from_ = "db"
    )

    return asset

def test( func, general_plot = False, **kwargs):

    asset = new()

    nasset = func(asset)

    t = kwargs.get("time", 1)
    
    nasset.df["target"] = nasset.df["close"].pct_change( t ).shift( t ).apply( lambda x: 1 if x > 0 else 0 )

    if general_plot:
        get_plot(nasset, **kwargs)

    return get_prob( nasset, **kwargs )

if __name__ != "__main__":

    df = pd.DataFrame()
    sizes = pd.DataFrame()

    for name in TREND_ONE_PARAM_NAMES:
    
        dff = test( trend , ta = name)

        df = pd.concat([df, dff[ "area" ] ], axis = 1)
        sizes = pd.concat([sizes, dff[ "largo" ] ], axis = 1)
    
    df.columns = TREND_ONE_PARAM_NAMES
    sizes.columns = TREND_ONE_PARAM_NAMES


