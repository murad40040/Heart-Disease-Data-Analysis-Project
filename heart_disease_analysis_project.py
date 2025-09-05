
# Heart Disease Data Analysis Project

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 2: Load the Dataset
df = pd.read_csv("heart.csv")
df.head()

# Step 3: List Data Attributes and Their Types
print("Data Types:\n", df.dtypes)

# Step 4: Display 10 Records
print("First 10 records:\n", df.head(10))

# -----------------------------
# Analysis 1: Classification
# -----------------------------

# Encode categorical features
df_encoded = df.copy()
label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# Evaluate
conf_matrix = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No HD", "HD"], yticklabels=["No HD", "HD"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.show()

# -----------------------------
# Analysis 2: Clustering
# -----------------------------

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

cluster_df = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
cluster_df["Cluster"] = clusters

# Cluster plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=cluster_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2")
plt.title("K-Means Clustering (k=2) via PCA")
plt.savefig("kmeans_pca.png")
plt.show()

# -----------------------------
# Analysis 3: Frequent Pattern Mining
# -----------------------------

# Prepare data
df_fp = df.copy()
df_fp['Age'] = pd.cut(df_fp['Age'], bins=3, labels=["Young", "Middle", "Old"])
df_fp['RestingBP'] = pd.cut(df_fp['RestingBP'], bins=3, labels=["LowBP", "NormalBP", "HighBP"])
df_fp['Cholesterol'] = pd.cut(df_fp['Cholesterol'], bins=3, labels=["LowChol", "MedChol", "HighChol"])
df_fp['MaxHR'] = pd.cut(df_fp['MaxHR'], bins=3, labels=["LowHR", "MedHR", "HighHR"])
df_fp['Oldpeak'] = pd.cut(df_fp['Oldpeak'], bins=3, labels=["LowOP", "MedOP", "HighOP"])

# Convert to transactions
transactions = df_fp.astype(str).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Run apriori
frequent_itemsets = apriori(df_trans, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter and show top rules
rules_hd = rules[rules['consequents'].astype(str).str.contains('HeartDisease')]
rules_hd_sorted = rules_hd.sort_values(by="lift", ascending=False).head(5)
print("Top Association Rules Involving HeartDisease:\n", rules_hd_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
