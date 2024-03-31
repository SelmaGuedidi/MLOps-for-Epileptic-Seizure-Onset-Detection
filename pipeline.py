
from tkinter import _test
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import randint
# Étape 1 : Initialisation du dépôt mlflow
mlflow.set_tracking_uri("http://127.0.0.1:8080")  # URL du serveur mlflow
#mlflow.set_experiment("epilepsy_detection")  # Nom de l'expérience mlflow

# Étape 2 : Définition de la source de données CSV
column_names = ["DFA_channel1", "Fisher_Information_channel1", "HFD_channel1", "PFD_channel1", "SVD_Entropy_channel1", "variance_channel1", "std_deviation_channel1", "mean_channel1", "fft_variance_channel1", "fft_std_deviation_channel1", "fft2_variance_channel1", "zero_crossing_rate_channel1", "complexity_channel1", "DFA_channel2", "Fisher_Information_channel2", "HFD_channel2", "PFD_channel2", "SVD_Entropy_channel2", "variance_channel2", "std_deviation_channel2", "mean_channel2", "fft_variance_channel2", "fft_std_deviation_channel2", "fft2_variance_channel2", "zero_crossing_rate_channel2", "complexity_channel2", "label"]
data = pd.read_csv("data.csv", header=None, names=column_names)

# Étape 3 : Fractionnement des données
X = data.drop(columns=['label'])
y = data['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "LogisticRegression": LogisticRegression(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "RandomForest_PCA": Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('model', RandomForestClassifier())]),
    "SVM_PCA": Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('model', SVC())]),
    "LogisticRegression_PCA": Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('model', LogisticRegression())]),
    "GradientBoosting_PCA": Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('model', GradientBoostingClassifier())]),
    "AdaBoost_PCA": Pipeline([('scaler', StandardScaler()), ('pca', PCA()), ('model', AdaBoostClassifier())]),
    #after doing some hyperparameter optimization
    
}

# Définition du pipeline avec GridSearchCV pour optimiser les hyperparamètres
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
     ('model', RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions={
            'pca__n_components': randint(5, 20),  # Varying PCA components
            'feature_selection__k': randint(5, 20),  # Varying number of features selected
            'model__n_estimators': randint(100, 1000),
            'model__max_depth': [None] + list(randint(3, 20).rvs(5)),
            'model__min_samples_split': randint(2, 11),
            'model__min_samples_leaf': randint(1, 11),
            'model__bootstrap': [True, False],
            'model__max_features': ['auto', 'sqrt', 'log2'],
            'model__class_weight': [None, 'balanced', 'balanced_subsample'],
            'model__criterion': ['gini', 'entropy']
        },
        cv=5,
        n_iter=200
    ))
])

# Boucle sur les modèles
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline.set_params(model=model)
        
        # Formation du pipeline
        pipeline.fit(X_train, y_train)
       
        y_pred = pipeline.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        test_accuracy = report['accuracy']
        test_recall = report['weighted avg']['recall']
        test_f1_score = report['weighted avg']['f1-score']
        
        print(f"Model: {model_name}, Accuracy on test: {test_accuracy}, Recall on test: {test_recall}, F1-score on test: {test_f1_score}")
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1_score)
        with open("results.txt", "a") as file:
            file.write(f"Model: {model_name}, Accuracy on test: {test_accuracy}, Recall on test: {test_recall}, F1-score on test: {test_f1_score}\n")
        # Exportation du pipeline
        mlflow.sklearn.log_model(pipeline, model_name)