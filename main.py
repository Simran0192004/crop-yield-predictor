import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load data
df = pd.read_csv('data/yield_df.csv').dropna()

# 2️⃣ Separate numeric and categorical features
num_feat = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
cat_feat = ['Area', 'Item']

X_num = df[num_feat]
X_cat = pd.get_dummies(df[cat_feat], drop_first=True)

# 3️⃣ Define target
y = df['hg/ha_yield']

# 4️⃣ Apply polynomial transformation to numeric features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_num_poly = poly.fit_transform(X_num)

# 5️⃣ Combine numeric + encoded categorical features
X_final = np.hstack([X_num_poly, X_cat.values])

# 6️⃣ Train your model
model = LinearRegression()
model.fit(X_final, y)

# 7️⃣ Predict using the same set of features
y_pred = model.predict(X_final)

# 8️⃣ Evaluate & Visualize
print("✅ MSE:", mean_squared_error(y, y_pred))
print("✅ R² :", r2_score(y, y_pred))

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.7, color='skyblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.show()
