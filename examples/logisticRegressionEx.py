from daedalus import Matrix, DataFrame, read_csv
from daedalus.model_selection import train_test_split
from daedalus.preprocessing import StandardScaler
from daedalus.models import LogisticRegression
from daedalus.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df: DataFrame = read_csv("ads.csv")
print(f"Loaded data: {df.rows} rows, {df.cols} columns")

df.encode_binary('Gender', "Male", "Female")

all_features = df.get_column_names()

features: list[str] = [feature for feature in all_features if feature not in ['Purchased', 'User ID']]
target = ['Purchased']

X_mat: Matrix = df.to_matrix(features)
y_mat: Matrix = df.to_matrix(target)
print("Feature Columns:", features)

X_train, X_test, y_train, y_test = train_test_split(
    X_mat, y_mat, test_size=0.2, seed=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LogisticRegression(learning_rate=0.001)
model.fit(X_train_scaled, y_train, epochs=500)

predictions: list = model.predict(X_test_scaled)
accuracy: float = accuracy_score(y_test, predictions)
recall: float = recall_score(y_test, predictions)
precision: float = precision_score(y_test, predictions)
conf_mat: Matrix = confusion_matrix(y_test, predictions)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix", conf_mat)
