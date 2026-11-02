"""Generate a realistic synthetic house dataset (prices in Lakh INR).
Replace data/house_data.csv with your own real dataset if you have one."""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500

locations = {"Anna Nagar": 1.35, "T Nagar": 1.45, "Velachery": 1.15,
             "Tambaram": 0.85, "OMR": 1.05, "Porur": 0.95}

area = rng.integers(600, 3500, N)
bedrooms = np.clip((area / 700 + rng.normal(0, 0.6, N)).round().astype(int), 1, 5)
bathrooms = np.clip((bedrooms + rng.integers(-1, 2, N)), 1, 5)
age = rng.integers(0, 31, N)
parking = rng.integers(0, 2, N)
location = rng.choice(list(locations), N)
loc_mult = np.array([locations[l] for l in location])

price = (area * 0.032 + bedrooms * 3 + bathrooms * 2.5 + parking * 4 - age * 0.6)
price = price * loc_mult + rng.normal(0, 3, N)
price = np.clip(price, 10, None).round(2)

df = pd.DataFrame({"Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
                   "Location": location, "Age": age, "Parking": parking,
                   "Price": price})
os.makedirs("data", exist_ok=True)
df.to_csv("data/house_data.csv", index=False)
print(df.head()); print("Saved", len(df), "rows")
