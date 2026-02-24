from daedalus import read_csv
from daedalus.model_selection import train_test_split
from daedalus.preprocessing import StandardScaler
from daedalus.models import LinearRegression
from daedalus.metrics import mean_squared_error

# Load Data
df = read_csv("boston_housing.csv")
print(f"Loaded data: {df.rows} rows, {df.cols} columns")

# Data Preparation
all_cols = df.get_column_names() 
features = [c for c in all_cols if c != 'medv']

X_mat = df.to_matrix(features)
y_mat = df.to_matrix(['medv'])

# Scale
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_mat)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_mat, test_size=0.2
)

# Train
model = LinearRegression(learning_rate=0.01)
model.fit(X_train, y_train, epochs=500)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Test MSE: {mse}")