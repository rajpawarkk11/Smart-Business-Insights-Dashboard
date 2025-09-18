import pandas as pd
import numpy as np
from datetime import date, timedelta

dates = pd.date_range(end=pd.Timestamp.today(), periods=120).to_series().dt.date
df = pd.DataFrame({
    "date": dates,
    "store_id": np.random.choice([1,2,3], size=len(dates)),
    "ad_spend": np.random.uniform(100,1000,size=len(dates)).round(2),
    "price": np.random.uniform(10,50,size=len(dates)).round(2),
    "units_sold": (np.random.poisson(lam=50, size=len(dates)) + np.random.randint(0,20,len(dates)))
})
df.to_csv("sample_sales.csv", index=False)
print("sample_sales.csv created")

