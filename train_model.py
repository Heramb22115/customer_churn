import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import time

TRAIN_FILE_PATH = "customer_churn_dataset-training-master.csv"
TEST_FILE_PATH = "customer_churn_dataset-testing-master.csv"

def load_and_preprocess_data(train_path, test_path):
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    def clean_df(df):
        df.dropna(inplace=True)
        cols_to_int = ['CustomerID', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                       'Payment Delay', 'Last Interaction', 'Churn']
        for col in cols_to_int:
            if col in df.columns:
                df[col] = df[col].astype('int64')
        df.reset_index(drop=True, inplace=True)
        return df

    df_train = clean_df(df_train)
    df_test = clean_df(df_test)

    y_train = df_train['Churn']
    X_train_data = df_train.drop(['CustomerID', 'Churn'], axis=1)

    y_test = df_test['Churn']
    X_test_data = df_test.drop(['CustomerID', 'Churn'], axis=1)

    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    
    combined_data = pd.concat([X_train_data, X_test_data], keys=['train', 'test'])
    combined_encoded = pd.get_dummies(combined_data, columns=categorical_cols, drop_first=True)
    
    X_train_encoded = combined_encoded.loc['train']
    X_test_encoded = combined_encoded.loc['test']
    
    numerical_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    
    scaler = StandardScaler()
    X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
    X_test_encoded[numerical_cols] = scaler.transform(X_test_encoded[numerical_cols])
    
    return X_train_encoded, X_test_encoded, y_train, y_test

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    end_time = time.time()
    
    print(f"\nModel: {model_name}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return model, roc_auc

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data(TRAIN_FILE_PATH, TEST_FILE_PATH)
    
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, n_jobs=-1, solver='saga', max_iter=500),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_model = None
    best_auc = -1

    for name, model in models.items():
        trained_model, auc = train_and_evaluate_model(model, X_train, y_train, X_test, y_test, name)
        results[name] = auc
        
        if auc > best_auc:
            best_auc = auc
            best_model = trained_model
            best_model_name = name

    print("\n--- FINAL MODEL COMPARISON (ROC AUC) ---")
    print(f"The best performing model is: {best_model_name} with ROC AUC: {best_auc:.4f}")
    