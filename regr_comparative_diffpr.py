# Generar el borrador para un ambiente donde se podran
# comparar diferentes regresiones

# No se tomaran variables extras, el objetivo sera ver como explotan
# la propia informacion historica del asset

# Para la prediccion, en vez de considerrse unicamente el target, se hara
# una ligera prediccion linear de las variables que tratamos, y con ellas hacer la prediccion 
# del chane en Close sin el shift como en target

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import date
import time
import json
import plotly.express as xp

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.neural_network import MLPRegressor

from trading.predictions import MyGridSearch
from trading.neuralnetworks import NN   
from trading.func_brokers import get_assets
from trading import Asset
from trading.processes import Simulation

STR_NAME = "comparison_results_newpred.json"

ERRORS = {
    "mae":mean_absolute_error,
    "mse":mean_squared_error
}

TARGETS = ["close_1", "target"]

def errors(yt, yp):

    results = {}
    for error_name, error in ERRORS.items():
        results[error_name] = error(yt, yp)

    return results

def normalize(df):

    for col in df.columns:
        df[col] = ( df[col] - df[col].min() ) / ( df[col].max() - df[col].min() )

    return df

def features(asset):
    """ Set asset df for further prediction 
    
        It will ensure time dependency
    """

    asset.df["rsi14"] = asset.rsi(14)
    asset.df["sma30"] = asset.sma(30)
    asset.df["ema30"] = asset.ema(30)
    asset.df["cci14"] = asset.cci(14)

    df = asset.df.copy()

    df["hl"] = df["high"] - df["low"]
    df["ho"] = df["high"] - df["open"]
    df["lo"] = df["low"] - df["open"]
    df["cl"] = df["close"] - df["low"]
    df["ch"] = df["close"] - df["high"]

    for i in [5, 30]:
        for c in ["close", "high", "volume"]:
            df["std{}_{}".format(c, i)] = df[c].rolling(i).std()

    cols = df.columns

    for c in cols:
        for i in range(1, 5):
            df["{}_{}".format(c, i)] = df[c].shift(i)
            df["{}_change_{}".format(c, i)] = df[c].pct_change(i)

    df["target"] = df["close"].pct_change(1).shift(-1)

    df = df.replace( [np.inf, -np.inf], np.nan ).dropna()

    df = normalize(df)

    train_len = int( len(df)*0.8 )

    train = df.iloc[ :train_len ]
    test = df.iloc[train_len:]

    return train, test

def lr(train, test):
    """ Linear Regression """
    
    results = {}
    results["params"] = {}

    start_time = time.time()
    r = []

    for target in TARGETS:
        regr = LinearRegression()

        drop = [target, "target"] if target != "target" else [target]

        regr.fit( train.drop(columns = drop), train[target])

        yp = regr.predict( test.drop(columns = drop)  )

        if target != "target":
            yp = [np.nan] + list(yp)[:-1]

        r.append( yp )

    yp = pd.DataFrame(r).T.dropna().mean(axis = 1).values
    
    end_time = time.time() - start_time
    
    results["time"] = end_time

    results["errors"] = errors( test["target"].iloc[1:], yp )

    return results

def log_r(train, test):
    """ Logistic Regression """
    params = {
        "penalty":["l1", "l2", "elasticnet", "none"],
        "tol":[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        "C":[100, 10, 1, 0.1, 0.01],
        "fit_intercept":[True, False],
        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter":[200, 100, 50]
    }
    
    r_total = []
    

    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        try:
            regr = LogisticRegression()

            for i, v in param.items(): regr.__dict__[i] = v

            regr.fit( train.drop(columns = "target"), train["target"])

            yp = regr.predict( test.drop(columns = "target")  )

            end_time = time.time() - start_time

            results["time"] = end_time
            results["errors"] = errors( test["target"], yp )
        
        except Exception as e:
            print("Error with params seting. \nException: {}".format(e))
            results["time"] = time.time() - start_time
            results["errors"] = None
            continue

        r_total.append( results )

    return r_total

def dt(train, test):
    """ Decision Tree """
    params = {
        "criterion":[ "poisson"],
        "splitter":["best"],
        "max_features":["auto", "sqrt", "log2", None]
    }
    
    r_total = []

    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        regr = DecisionTreeRegressor()

        for i, v in param.items(): regr.__dict__[i] = v

        r = []

        for target in TARGETS:
            
            drop = [target, "target"] if target != "target" else [target]

            regr.fit( train.drop(columns = drop), train[target])

            yp = regr.predict( test.drop(columns = drop)  )

            if target != "target":
                yp = [np.nan] + list(yp)[:-1]

            r.append( yp )

        yp = pd.DataFrame(r).T.dropna().mean(axis = 1).values

        end_time = time.time() - start_time

        results["time"] = end_time
        results["errors"] = errors( test["target"].iloc[1:], yp )

        r_total.append( results )

    return r_total

def mlp(train, test):
    """ Multilayer Perceptron """
    params = {
        "hidden_layer_sizes":[10, 20, 50],
        "activation":[ "logistic"],
        "solver":["adam"],
        "alpha":[ 0.0001, 0.00001, 0.000001],
        "learning_rate":["constant"],
        "learning_rate_init":[0.0001],
        "max_iter":[100],
        "tol":[0.0001]
    }
    
    r_total = []
    
    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        try:
            regr = MLPRegressor()

            for i, v in param.items(): regr.__dict__[i] = v

            r = []

            for target in TARGETS:
                
                drop = [target, "target"] if target != "target" else [target]

                regr.fit( train.drop(columns = drop), train[target])

                yp = regr.predict( test.drop(columns = drop)  )

                if target != "target":
                    yp = [np.nan] + list(yp)[:-1]

                r.append( yp )

            yp = pd.DataFrame(r).T.dropna().mean(axis = 1).values

            end_time = time.time() - start_time

            results["time"] = end_time
            results["errors"] = errors( test["target"].iloc[1:], yp )

        except Exception as e:
            print("Error with params seting. \nException: {}".format(e))
            results["time"] = time.time() - start_time
            results["errors"] = None
            continue

        r_total.append( results )

    return r_total

def lstm(train, test):
    """ Long-Short Term memory """
    
    params = {
        "nb_epoch":[100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        "neurons":[3,5,7,9]
    }
    
    r_total = []
    

    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        nn = NN()

        train_s, test_s = nn.scale( train, test )

        nn.fit( train_s, nb_epoch = param["nb_epoch"], neurons = param["neurons"])

        yp = nn.predict(test_s, batch_size = 1)

        end_time = time.time() - start_time

        results["time"] = end_time
        results["errors"] = errors( test["target"], yp )

        r_total.append( results )

    return r_total

def svm(train, test):
    """ Support vector Regression """
    params = {
        "kernel":[ "rbf", "sigmoid"],
        "gamma":[ "auto"],
        "tol":[0.01, 0.001, 0.0001, 0.00001],
        "C":[ 0.1, 0.01, 0.001],
        "coef0":[10, 1, 0.1],
        "epsilon":[0.5, 0.1, 0.001],
        "shrinking":[True, False]
    }
    
    r_total = []

    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        try:
            regr = SVR()

            for i, v in param.items(): regr.__dict__[i] = v

            r = []

            for target in TARGETS:
                
                drop = [target, "target"] if target != "target" else [target]

                regr.fit( train.drop(columns = drop), train[target])

                yp = regr.predict( test.drop(columns = drop)  )

                if target != "target":
                    yp = [np.nan] + list(yp)[:-1]

                r.append( yp )

            yp = pd.DataFrame(r).T.dropna().mean(axis = 1).values

            end_time = time.time() - start_time

            results["time"] = end_time
            results["errors"] = errors( test["target"].iloc[1:], yp )

        except Exception as e:
            print("Error with params seting. \nException: {}".format(e))
            results["time"] = time.time() - start_time
            results["errors"] = None
            continue

        r_total.append( results )

    return r_total

def rf(train, test):
    """ Random Forest """
    params = {
        "n_estimators":[10, 20, 50, 100, 200, 500],
        "criterion":["poisson"],
        "bootstrap":[False],

    }
    
    r_total = []

    for param in list( ParameterGrid( params ) ):    

        start_time = time.time()

        print(param)

        results = {}
        results["params"] = param

        regr = RandomForestRegressor()

        for i, v in param.items(): regr.__dict__[i] = v

        r = []

        for target in TARGETS:
            
            drop = [target, "target"] if target != "target" else [target]

            regr.fit( train.drop(columns = drop), train[target])

            yp = regr.predict( test.drop(columns = drop)  )

            if target != "target":
                yp = [np.nan] + list(yp)[:-1]

            r.append( yp )

        yp = pd.DataFrame(r).T.dropna().mean(axis = 1).values

        end_time = time.time() - start_time

        results["time"] = end_time
        results["errors"] = errors( test["target"].iloc[1:], yp )

        r_total.append( results )

    return r_total

def main():

    ALGORITHMS = {
        # "lr":lr,
        "dt":dt,
        "rf":rf,
        # "log_r":log_r,
        # "lstm":lstm,
        "mlp":mlp,
        "svm":svm
    }

    try:
        with open(STR_NAME, "r") as fp:
            data = json.load(fp)
    except Exception as e:
        print(e)
        data = {}

    assets = get_assets()["binance"]

    for i in list( assets.keys() )[:1]:
        asset_time = time.time()
        print(i)

        if i not in data:
            data[i] = {}
        
        asset = Asset(
            symbol=i,
            start = date( 2020,1,1 ),
            end = date(2022,3,1),
            frequency="1d",
            broker = "binance",
            fiat = "usdt",
            from_ = "db"
        )

        if asset.df is None or len(asset.df) == 0:
            continue

        train, test = features(asset)

        for alg_name, alg in ALGORITHMS.items():
            print(alg_name)
            alg_time = time.time()

            data[i][alg_name] = alg(train.copy(), test.copy())

            print("Algorithm time: ", time.time()- alg_time)

            # if brakes
            try:
                with open(STR_NAME, "w") as fp:
                    json.dump(data, fp)
            except Exception as e:
                print(e)
                print(data)
        
        print( "Asset time: ", time.time()- asset_time )

    return data

def unpack(df):

    df = pd.DataFrame(df)

    df["mae"] = df["errors"].apply( lambda x : x["mae"] )
    df["mse"] = df["errors"].apply( lambda x : x["mse"] )

    df.sort_values(by = "mse", ascending=True, inplace=True)

    return df

def analyze():
    try:
        with open("../" + STR_NAME, "r") as fp:
            data = json.load(fp)
    except Exception as e:
        print(e)
        data = {}
    
    exclude = ["lr"]

    data = data["ADA"]

    print( data.keys() )

    for d in data.keys():
        if d in exclude: continue

        print(d)

        if len(data[d]) == 0: continue

        df = unpack(data[d])


        print(df.head(3)[["time", "mae", "mse"]])


    fig = xp.scatter( df, x = "time", y =  ["mae", "mse"], hover_data = ["params"])
    fig.show()

    


def test():
    pass

if __name__ == "__main__":
    
    data = main()

    try:
        with open(STR_NAME, "w") as fp:
            json.dump(data, fp)
    except Exception as e:
        print(e)
        print(data)


 