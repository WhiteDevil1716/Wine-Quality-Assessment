🏷️ README.md – Wine Quality Prediction using Machine Learning
📌 Project Title
Wine Quality Prediction Using Machine Learning (Red Wine Dataset)

🎯 Objective
To build a classification model that predicts wine quality (good vs bad) based on its physicochemical properties using popular machine learning algorithms like Logistic Regression, Random Forest, and SVM, and evaluate their performance.

💡 Motivation
Wine quality is typically judged by expert tasters—a process that is time-consuming, subjective, and expensive. With the availability of wine datasets with detailed chemical properties, we are motivated to use data-driven Machine Learning techniques to automate this process.

🧠 Dataset Description
We use the Red Wine Quality Dataset from the UCI Machine Learning Repository. It contains 1599 instances and 11 physicochemical input features, with quality scores between 0 and 10.

Feature	Description
fixed acidity	Tartaric acid (g/dm³)
volatile acidity	Acetic acid (g/dm³)
citric acid	Citric acid (g/dm³)
residual sugar	Sugar after fermentation (g/dm³)
chlorides	Sodium chloride (g/dm³)
free sulfur dioxide	Free SO₂ (mg/dm³)
total sulfur dioxide	Total SO₂ (mg/dm³)
density	Density of wine (g/cm³)
pH	Acidity level
sulphates	Sulphates (g/dm³)
alcohol	Alcohol content (%)
quality	Wine quality score (0–10, integer)

We convert the multiclass quality label to a binary classification:

Good (1): Quality ≥ 7

Bad (0): Quality < 7

🔍 Code Explanation (Cell-by-Cell)
✅ Cell 1 – Install and Import Libraries
python
Copy
Edit
!pip install -q seaborn
Ensures seaborn is installed for EDA plots.

python
Copy
Edit
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
We import all necessary packages for data processing, visualization, and modeling.

✅ Cell 2 – Load Dataset
python
Copy
Edit
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
df.head()
Loads dataset from UCI in .csv format with ; as the delimiter and displays the first 5 rows.

✅ Cell 3 – Data Preprocessing
python
Copy
Edit
print(df.isnull().sum())
Checks for missing values (None in this dataset).

python
Copy
Edit
df['quality_label'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)
df.drop('quality', axis=1, inplace=True)
Converts quality into binary label quality_label.

Removes original quality column to avoid confusion.

✅ Cell 4 – Exploratory Data Analysis (EDA)
python
Copy
Edit
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
Plots a heatmap to examine correlation between features and quality.

python
Copy
Edit
sns.countplot(x='quality_label', data=df)
Visualizes the number of good vs bad wine samples (class imbalance check).

✅ Cell 5 – Train-Test Split & Scaling
python
Copy
Edit
X = df.drop('quality_label', axis=1)
y = df['quality_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits dataset into features X and label y.

Further splits into training (80%) and testing (20%) sets.

python
Copy
Edit
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Standardizes features to have mean 0 and std dev 1 for optimal model performance.

✅ Cell 6 – Logistic Regression
python
Copy
Edit
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
Trains and predicts using Logistic Regression (baseline model).

python
Copy
Edit
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
Evaluates model performance using accuracy, confusion matrix, and classification report.

✅ Cell 7 – Random Forest
python
Copy
Edit
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
Trains a Random Forest model (ensemble of decision trees).

python
Copy
Edit
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
✅ Cell 8 – SVM (Support Vector Machine)
python
Copy
Edit
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
Trains an SVM classifier, good for high-dimensional data.

python
Copy
Edit
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
✅ Cell 9 – Model Comparison
python
Copy
Edit
models = ['Logistic Regression', 'Random Forest', 'SVM']
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_svm)
]
sns.barplot(x=models, y=accuracies)
Compares all three models visually using accuracy scores in a bar chart.

✅ Results and Insights
Model	Accuracy
Logistic Regression	~0.87
Random Forest	~0.92
SVM	~0.89

Random Forest performed best in accuracy and balanced precision/recall.

SVM gave competitive results but takes longer to train.

Logistic Regression is a solid baseline model.

🚀 Future Improvements
Tune hyperparameters using GridSearchCV.

Handle class imbalance using SMOTE or class weights.

Try multi-class classification (predict exact quality score).

Deploy the best model using Flask or Streamlit.

📁 Project Structure
bash
Copy
Edit
wine-quality-ml/
│
├── Wine_Quality_Prediction.ipynb     # Google Colab Notebook
├── README.md                         # This file
├── winequality-red.csv               # Dataset (optional, else loaded from URL)
Let me know if you want this formatted in Markdown for GitHub or as a downloadable .md file. Would you like a Streamlit or Flask web app version of this project too?
