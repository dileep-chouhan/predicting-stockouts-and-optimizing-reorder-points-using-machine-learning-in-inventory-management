import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_days = 365
data = {
    'Date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=num_days)),
    'Sales': np.random.randint(50, 200, size=num_days) + np.sin(np.linspace(0, 2*np.pi, num_days)) * 50, #Seasonal sales pattern
    'LeadTime': np.random.randint(2, 7, size=num_days), #Lead time in days
    'Price': np.random.uniform(10, 25, size=num_days) #Product price
}
df = pd.DataFrame(data)
#Add some external factors (noise for simplicity)
df['ExternalFactor1'] = np.random.normal(0, 5, size=num_days)
df['ExternalFactor2'] = np.random.normal(1, 2, size=num_days)
# --- 2. Feature Engineering ---
df['RollingAvgSales'] = df['Sales'].rolling(window=30).mean() #30-day rolling average sales
df['LaggedSales'] = df['Sales'].shift(1) #Yesterday's sales
df.dropna(inplace=True) #remove rows with NaN after lag
# --- 3. Model Training (Linear Regression for simplicity) ---
X = df[['RollingAvgSales', 'LaggedSales', 'LeadTime', 'Price', 'ExternalFactor1', 'ExternalFactor2']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# --- 4. Prediction and Reorder Point Calculation ---
#Example:  Reorder point = predicted sales during lead time + safety stock
predictions = model.predict(X_test)
df_test = pd.DataFrame({'Date':df['Date'][len(X_train):], 'PredictedSales': predictions})
df_test['LeadTimeSales'] = df_test['PredictedSales'] * df['LeadTime'][len(X_train):] #Sales during lead time
df_test['SafetyStock'] = 10 #Example safety stock - needs more sophisticated calculation in real scenarios
df_test['ReorderPoint'] = df_test['LeadTimeSales'] + df_test['SafetyStock']
# --- 5. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(df_test['Date'], df_test['PredictedSales'], label='Predicted Sales')
plt.plot(df_test['Date'], df_test['ReorderPoint'], label='Reorder Point')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Predicted Sales and Reorder Points')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename = 'reorder_point_prediction.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10,6))
sns.regplot(x=y_test, y=predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
output_filename2 = 'actual_vs_predicted.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")