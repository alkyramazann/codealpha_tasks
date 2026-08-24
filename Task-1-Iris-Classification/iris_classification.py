import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


sns.set_style("whitegrid")


print("=" * 60)
print("STEP 1: LOAD AND INSPECT THE DATA")
print("=" * 60)


df = pd.read_csv("Iris.csv")


print("\nFirst 5 rows of the dataset:")
print(df.head())


print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")


print("\nColumn names and data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

df = df.drop(columns=["Id"])

target_column = "Species"
feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

print(f"\nTarget column: '{target_column}'")
print(f"Feature columns: {feature_columns}")
print(f"\nClass balance (rows per species):")
print(df[target_column].value_counts())


print("\n" + "=" * 60)
print("STEP 2: PREPARE THE DATA")
print("=" * 60)


X = df[feature_columns]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled = scaler.transform(X_test) 

print("\nFeature scaling applied (StandardScaler): each feature now has")
print("mean = 0 and standard deviation = 1.")



print("\n" + "=" * 60)
print("STEP 3: BUILD THE MODEL")
print("=" * 60)


model = KNeighborsClassifier(n_neighbors=5)


model.fit(X_train_scaled, y_train)
print("Model trained: K-Nearest Neighbors (KNN) with k = 5")



print("\n" + "=" * 60)
print("STEP 4: EVALUATE THE MODEL")
print("=" * 60)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

labels_order = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
print("\nConfusion Matrix:")
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual: {l}" for l in labels_order],
    columns=[f"Pred: {l}" for l in labels_order],
)
print(cm_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels_order))



print("\n" + "=" * 60)
print("STEP 5: VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
disp.plot(ax=axes[0], cmap="Blues", colorbar=False, xticks_rotation=30)
axes[0].set_title("Confusion Matrix (Test Data)")


species_colors = {
    "Iris-setosa": "tab:blue",
    "Iris-versicolor": "tab:orange",
    "Iris-virginica": "tab:green",
}
for species, color in species_colors.items():
    subset = df[df["Species"] == species]
    axes[1].scatter(
        subset["PetalLengthCm"],
        subset["PetalWidthCm"],
        label=species,
        color=color,
        alpha=0.7,
        edgecolor="k",
    )
axes[1].set_xlabel("Petal Length (cm)")
axes[1].set_ylabel("Petal Width (cm)")
axes[1].set_title("Petal Length vs Petal Width by Species")
axes[1].legend()

plt.tight_layout()
plt.savefig("iris_results.png", dpi=150)
print("\nVisualizations saved as 'iris_results.png'")



results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
})
misclassified = results_df[results_df["Actual"] != results_df["Predicted"]]
print("\n" + "=" * 60)
print("MISCLASSIFIED SAMPLES (if any):")
print("=" * 60)
if len(misclassified) == 0:
    print("None — every test sample was classified correctly!")
else:
    print(misclassified)

print("\nDone.")
