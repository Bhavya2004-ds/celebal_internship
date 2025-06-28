#Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Step 2: Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

#Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Train basic models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

print("Basic Model Performance:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print()

#Step 5: GridSearchCV on Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best Random Forest Parameters:", grid.best_params_)

#Step 6: RandomizedSearchCV on SVM
param_dist = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

random_search = RandomizedSearchCV(SVC(), param_dist, n_iter=5, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)
print("Best SVM Parameters:", random_search.best_params_)

#Step 7: Final evaluation of best models
print("\nFinal Evaluation on Test Set:\n")

# Best Random Forest
best_rf = grid.best_estimator_
rf_pred = best_rf.predict(X_test)
print("Random Forest Report:\n", classification_report(y_test, rf_pred))

#Best SVM
best_svm = random_search.best_estimator_
svm_pred = best_svm.predict(X_test)
print("SVM Report:\n", classification_report(y_test, svm_pred))
