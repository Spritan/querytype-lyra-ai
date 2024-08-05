import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import numpy as np

# Load data
df = pd.read_csv('./data/syn.csv')

# Encode labels
label_encoder = LabelEncoder()
df['answer'] = label_encoder.fit_transform(df['answer'])

# Split data into features and labels
X_train, X_test, y_train, y_test = train_test_split(df['query'], df['answer'], test_size=0.2, random_state=42)

# Define vectorizers and models
vectorizers = {
    'Bag of Words': CountVectorizer(),
    'TF-IDF': TfidfVectorizer(),
    'Hashing': HashingVectorizer(n_features=10000)  # Hashing vectorizer with a fixed number of features
}

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'CatBoost': CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100)
}

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

# Create a DataFrame to store results
results = []

# Initialize variables to keep track of the best model and vectorizer
best_f1_score = -1
best_vectorizer_name = None
best_model_name = None
best_vectorizer = None
best_model = None

# Measure the start time
start_time = time.time()

for vectorizer_name, vectorizer in vectorizers.items():
    for model_name, model in models.items():
        if vectorizer_name == 'Hashing' and model_name == 'Naive Bayes':
            print(f"Skipping {vectorizer_name} with {model_name} due to incompatible data type.\n")
            continue
        
        print(f"Using {vectorizer_name} with {model_name}:\n")
        
        # Create and train the pipeline
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Convert the data type for LightGBM
        if model_name == 'LightGBM':
            X_train_vec = X_train_vec.astype(np.float32)
            X_test_vec = X_test_vec.astype(np.float32)
        
        # Handle hashing vectorizer output with scaling for some models
        if vectorizer_name == 'Hashing':
            scaler = StandardScaler(with_mean=False)
            X_train_vec = scaler.fit_transform(X_train_vec)
            X_test_vec = scaler.transform(X_test_vec)
        
        model.fit(X_train_vec, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store the classification report
        for label, metrics in report.items():
            if label != 'accuracy':  # Skip accuracy as it's not a class-specific metric
                results.append({
                    'Vectorizer': vectorizer_name,
                    'Model': model_name,
                    'Class': label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        # Calculate average F1-score for this model and vectorizer combination
        f1_scores = [metrics['f1-score'] for label, metrics in report.items() if label != 'accuracy']
        avg_f1_score = np.mean(f1_scores)
        
        # Check if this is the best performing model
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_vectorizer_name = vectorizer_name
            best_model_name = model_name
            best_vectorizer = vectorizer
            best_model = model
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, f'{vectorizer_name} + {model_name} Confusion Matrix', f'{vectorizer_name}_{model_name}_confusion_matrix.png')

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

# Save the best model and vectorizer
if best_vectorizer and best_model:
    joblib.dump(best_vectorizer, 'best_vectorizer.pkl')
    joblib.dump(best_model, 'best_model.pkl')
    print(f"Best model and vectorizer have been saved as 'best_model.pkl' and 'best_vectorizer.pkl'.")

# Measure the end time
end_time = time.time()

# Calculate and print the total elapsed time
total_elapsed_time = end_time - start_time
print(f"Total execution time: {total_elapsed_time:.4f} seconds")

print("Evaluation results have been saved to 'model_evaluation_results.csv'.")
