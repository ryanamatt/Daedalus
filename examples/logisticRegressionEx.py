from daedalus import Matrix, DataFrame, read_csv
from daedalus.model_selection import train_test_split
from daedalus.preprocessing import StandardScaler
from daedalus.models import LogisticRegression
from daedalus.metrics import accuracy_score

df: DataFrame = read_csv("ads.csv")
print(f"Loaded data: {df.rows} rows, {df.cols} columns")

all_features = df.get_column_names()

features: list[str] = [feature for feature in all_features if feature != 'Purchased']
target = ['Purchased']

X_mat: Matrix = df.to_matrix(features)
y_mat: Matrix = df.to_matrix(target)
print("Feature Columns:", features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mat)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_mat, test_size=0.2
)

model = LogisticRegression(learning_rate=0.01)
model.fit(X_train, y_train, epochs=500)

predictions: list = model.predict(X_test)
accuracy: float = accuracy_score(y_test, predictions)

print("Predictions Accuracy:", accuracy)
