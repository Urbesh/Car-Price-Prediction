
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(file_path):
    print(f"Reading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Check for required columns
    required_columns = ['Predicted_Price', 'Actual_Price']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in the CSV file.")
            print(f"Available columns: {df.columns.tolist()}")
            return

    # Drop rows with missing values in predicted or actual price
    df_clean = df.dropna(subset=required_columns)
    
    # Ensure data is numeric
    # It seems there might be some non-numeric characters based on my brief view (though it looked mostly numeric)
    # Using 'coerce' to turn errors into NaNs and then dropping them
    df_clean['Predicted_Price'] = pd.to_numeric(df_clean['Predicted_Price'], errors='coerce')
    df_clean['Actual_Price'] = pd.to_numeric(df_clean['Actual_Price'], errors='coerce')
    df_clean = df_clean.dropna(subset=required_columns)

    y_pred = df_clean['Predicted_Price']
    y_true = df_clean['Actual_Price']

    # 1. Calculate Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("-" * 30)
    print("Model Performance Metrics:")
    print("-" * 30)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("-" * 30)

    # 2. Residual Analysis
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    print("\nResidual Analysis:")
    print("-" * 30)
    print(f"Mean Residual: {mean_residual:.2f} (Should be close to 0)")
    print(f"Std Dev of Residuals: {std_residual:.2f}")
    print(f"Min Residual: {np.min(residuals):.2f}")
    print(f"Max Residual: {np.max(residuals):.2f}")
    
    # Check for large errors (Outliers)
    # Let's consider an error > 2 * RMSE as a significant outlier
    threshold = 2 * rmse
    outliers = df_clean[np.abs(residuals) > threshold]
    print(f"\nNumber of significant outliers (Error > 2*RMSE): {len(outliers)} out of {len(df_clean)} samples ({len(outliers)/len(df_clean)*100:.2f}%)")

    # Sample some large outliers to see what's going on
    if not outliers.empty:
        print("\nTop 5 Largest Positive Errors (Underprediction - Actual > Predicted):")
        top_pos_outliers = outliers[residuals > 0].nlargest(5, 'Actual_Price') # Show based on high actual price? or high error?
        # Let's sort outliers by absolute residual
        outliers['Abs_Residual'] = outliers.apply(lambda row: abs(row['Actual_Price'] - row['Predicted_Price']), axis=1)
        top_outliers = outliers.sort_values(by='Abs_Residual', ascending=False).head(5)
        
        
        print(top_outliers[['Manufacturer', 'Model', 'Actual_Price', 'Predicted_Price']].to_string())

    # 3. Cleaned Analysis (No extreme outliers)
    print("\n" + "=" * 30)
    print("Metrics on Cleaned Data (Removing Actual_Price > 200,000)")
    print("=" * 30)
    
    # Remove rows where Actual Price is excessive (likely data entry errors like 26M)
    df_clean_filtered = df_clean[df_clean['Actual_Price'] < 200000]
    
    if len(df_clean_filtered) < len(df_clean):
        y_pred_clean = df_clean_filtered['Predicted_Price']
        y_true_clean = df_clean_filtered['Actual_Price']
        
        mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
        mse_clean = mean_squared_error(y_true_clean, y_pred_clean)
        rmse_clean = np.sqrt(mse_clean)
        r2_clean = r2_score(y_true_clean, y_pred_clean)
        
        print(f"Removed {len(df_clean) - len(df_clean_filtered)} extreme outliers.")
        print(f"New MAE: {mae_clean:.2f}")
        print(f"New RMSE: {rmse_clean:.2f}")
        print(f"New R2 Score: {r2_clean:.4f}")
    else:
        print("No extreme outliers (> 200,000) found.")

if __name__ == "__main__":
    file_path = r"c:\Users\urbes\Desktop\Ml1\Car-Price-Prediction\Full Project\Output\Predicted_price.csv"
    evaluate_model(file_path)
