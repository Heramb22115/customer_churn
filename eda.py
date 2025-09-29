import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_FILE_PATH = "customer_churn_dataset-training-master.csv"

def clean_data_for_eda(df):
    
    df.dropna(inplace=True)
    
    cols_to_int = ['CustomerID', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                   'Payment Delay', 'Last Interaction', 'Churn']
    
    for col in cols_to_int:
        if col in df.columns:
            
            df[col] = df[col].astype('int64')
            
    df.reset_index(drop=True, inplace=True)
    return df

def perform_eda(df):
    """Performs the main exploratory data analysis and saves plots."""
    print("--- Starting Exploratory Data Analysis (EDA) ---")
   
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution (0=No Churn, 1=Churn)')
    plt.savefig('eda_churn_distribution.png')
    plt.close()
    print("Generated: eda_churn_distribution.png")

    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(categorical_cols):
       
        churn_rate = df.groupby(col)['Churn'].mean().reset_index()
        sns.barplot(x=col, y='Churn', data=churn_rate, ax=axes[i], palette='viridis')
        axes[i].set_title(f'2. Churn Rate by {col}')
        axes[i].set_ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig('eda_categorical_churn_rate.png')
    plt.close()
    print("Generated: eda_categorical_churn_rate.png")

    numerical_cols = ['Tenure', 'Total Spend', 'Age']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, col in enumerate(numerical_cols):
        sns.kdeplot(data=df, x=col, hue='Churn', fill=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col} by Churn Status')
    plt.tight_layout()
    plt.savefig('eda_numerical_distributions.png')
    plt.close()
    print("Generated: eda_numerical_distributions.png")

    plt.figure(figsize=(10, 8))
    
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.drop('CustomerID', axis=1).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('eda_correlation_matrix.png')
    plt.close()
    print("Generated: eda_correlation_matrix.png")
    
    print("--- EDA Complete. Check the generated PNG files for insights. ---")

if __name__ == "__main__":
    df_train = pd.read_csv(TRAIN_FILE_PATH)
    df_train_clean = clean_data_for_eda(df_train.copy())
    perform_eda(df_train_clean)