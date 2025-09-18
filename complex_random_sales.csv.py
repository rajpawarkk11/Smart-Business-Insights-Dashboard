import pandas as pd
import numpy as np
from datetime import datetime, timedelta

num_rows = 200
store_ids = range(1, 11)
regions = ['North', 'South', 'East', 'West']
promotions = ['None', 'Email', 'Ads', 'Combo']

dates = [datetime.today().date() - timedelta(days=i) for i in range(num_rows)]
df = pd.DataFrame({
    'date': dates,
    'store_id': np.random.choice(store_ids, size=num_rows),
    'region': np.random.choice(regions, size=num_rows),
    'category': np.random.choice(['Electronics', 'Clothing', 'Grocery', 'Books'], size=num_rows),
    'day_of_week': [d.strftime('%A') for d in dates],
    'ad_spend': np.random.uniform(100, 2000, size=num_rows).round(2),
    'price': np.random.uniform(10, 200, size=num_rows).round(2),
    'discount_percent': np.random.uniform(0, 30, size=num_rows).round(2),
    'units_sold': np.random.poisson(lam=50, size=num_rows) + np.random.randint(0,20,num_rows),
    'online_sales': np.random.randint(0, 50, size=num_rows),
    'promotion': np.random.choice(promotions, size=num_rows),
    'customer_rating': np.clip(np.random.normal(4,0.5,num_rows),1,5).round(1)
})

df.to_csv("complex_random_sales.csv", index=False)
print("complex_random_sales.csv created successfully!")
