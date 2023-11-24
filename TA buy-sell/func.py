from trading import Asset
from datetime import date

def new_day():
    """ Get asset auxiliar function """
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

def new_week():
    """ Get asset auxiliar function """
    asset = Asset(
        symbol = "AAPL",
        broker="gbm",
        fiat = "mx",
        frequency="1w",
        start = date(2012,1,1),
        end = date(2018,1,1),
        from_ = "db"
    )

    return asset

def new_month():
    """ Get asset auxiliar function """
    asset = Asset(
        symbol = "AAPL",
        broker="gbm",
        fiat = "mx",
        frequency="1m",
        start = date(2017,1,1),
        end = date(2021,12,1),
        from_ = "db"
    )

    return asset