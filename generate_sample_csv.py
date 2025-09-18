import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------- Parameters ----------------
num_rows = 150  # Number of rows
num_stores = 5  # Number of unique store IDs
categories = ['Electronics', 'Clothing', 'Grocery', 'Books']  # Example categories

# ---------------- Generate Data ----------------
dates = [datetime.today().date() - timedelta(days=i) for i in range(num_rows)]
store_ids = np.random.choice(range(1, num_stores+1), size=num_rows)
categories_col = np.random.choice(categories, size=num_rows)
ad_spend = np.random.uniform(100, 1000, size=num_rows).round(2)
price = np.random.uniform(10, 100, size=num_rows).round(2)
units_sold = np.random.poisson(lam=50, size=num_rows) + np.random.randint(0, 20, size=num_rows)

# ---------------- Create DataFrame ----------------
df = pd.DataFrame({
    'date': dates,
    'store_id': store_ids,
    'category': categories_col,
    'ad_spend': ad_spend,
    'price': price,
    'units_sold': units_sold
})

# ---------------- Save CSV ----------------
file_name = "random_sample_sales.csv"
df.to_csv(file_name, index=False)
print(f"{file_name} created successfully with {num_rows} rows!")
