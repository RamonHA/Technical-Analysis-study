from trading import Asset
from datetime import date

def new():
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