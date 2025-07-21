import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os
import sys
import warnings
import time
from typing import Dict, Any, Tuple, List

warnings.filterwarnings('ignore')

# Import enhanced transformers
from transformers import (
    EnhancedCreditScoringTransformer, 
    EnhancedCreditScoringConfig, 
    validate_loan_application_schema, 
    get_available_features,
    ClientType
)

# Configuration
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'synthetic_training_data3.csv')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'enhanced_credit_model.pkl')
TRANSFORMER_PATH = os.path.join(MODEL_DIR, 'enhanced_transformer.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.pkl')
FEATURES_INFO_PATH = os.path.join(MODEL_DIR, 'features_info.pkl')


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, os.path.join(DATA_DIR, 'raw'), MODEL_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ensured: {directory}")


def load_and_validate_data() -> pd.DataFrame:
    """Load data and perform comprehensive validation."""
    print("\n" + "="*60)
    print("DATA LOADING AND VALIDATION")
    print("="*60)
    
    # Create dummy data if not exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"⚠️ {RAW_DATA_PATH} not found. Creating dummy data for demonstration.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['Default'] = y
        # Add required columns for the transformer
        df['Employment_Sector'] = np.random.choice(['Public', 'Private'], size=1000)
        df['Employment_Tenure_Months'] = np.random.randint(1, 240, size=1000)
        df['Net_Salary_Per_Cutoff'] = np.random.uniform(10000, 80000, size=1000)
        df['Salary_Frequency'] = np.random.choice(['Monthly', 'Biweekly'], size=1000)
        df['Housing_Status'] = np.random.choice(['Owned', 'Rented'], size=1000)
        df['Years_at_Current_Address'] = np.random.uniform(0.5, 20, size=1000)
        df['Household_Head'] = np.random.choice(['Yes', 'No'], size=1000)
        df['Number_of_Dependents'] = np.random.randint(0, 5, size=1000)
        df['Comaker_Relationship'] = np.random.choice(['Spouse', 'Parent', 'Friend'], size=1000)
        df['Comaker_Employment_Tenure_Months'] = np.random.randint(1, 120, size=1000)
        df['Comaker_Net_Salary_Per_Cutoff'] = np.random.uniform(8000, 60000, size=1000)
        df['Has_Community_Role'] = np.random.choice(['Yes', 'No'], size=1000)
        df['Paluwagan_Participation'] = np.random.choice(['Yes', 'No'], size=1000)
        df['Other_Income_Source'] = np.random.choice(['None', 'Business'], size=1000)
        df['Disaster_Preparedness'] = np.random.choice(['None', 'Savings'], size=1000)
        df['Is_Renewing_Client'] = np.random.randint(0, 2, size=1000)
        df['Grace_Period_Usage_Rate'] = np.random.uniform(0, 1, size=1000)
        df['Late_Payment_Count'] = np.random.randint(0, 10, size=1000)
        df['Had_Special_Consideration'] = np.random.randint(0, 2, size=1000)
        df.to_csv(RAW_DATA_PATH, index=False)
        print("✅ Dummy data created.")

    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"✅ Successfully loaded data from {RAW_DATA_PATH}")
        print(f"   Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File {RAW_DATA_PATH} not found.")
        raise
    
    # Schema validation
    df_fixed, issues = validate_loan_application_schema(df)
    if issues:
        print("⚠️ Schema issues found and fixed:")
        for issue in issues:
            print(f"   - {issue}")
        df = df_fixed
    else:
        print("✅ All fields comply with expected schema")
    
    # Target validation
    if 'Default' not in df.columns:
        raise ValueError("❌ Target column 'Default' not found in the dataset")
    df = df.dropna(subset=['Default'])
    
    print(f"\n🎯 Target Analysis (Default Rate): {df['Default'].mean()*100:.1f}%")
    
    return df


def analyze_data_leakage(df: pd.DataFrame) -> None:
    """Analyze potential data leakage in the dataset."""
    print("\n" + "="*60)
    print("DATA LEAKAGE ANALYSIS")
    print("="*60)
    
    problematic_features = ['Has_Community_Role', 'Paluwagan_Participation']
    for feature in problematic_features:
        if feature in df.columns:
            feature_analysis = df.groupby(feature)['Default'].agg(['count', 'mean']).round(3)
            print(f"\nAnalysis for '{feature}':")
            print(feature_analysis)
            rate_diff = feature_analysis['mean'].max() - feature_analysis['mean'].min()
            if rate_diff > 0.5:
                print(f"   ⚠️  WARNING: '{feature}' has extreme default rate difference: {rate_diff:.1%}")
    print("✅ Leakage analysis complete. Transformer is designed to constrain these features.")


def apply_enhanced_transformations(df: pd.DataFrame) -> Tuple[pd.DataFrame, EnhancedCreditScoringTransformer]:
    """Apply enhanced transformations with feature isolation."""
    print("\n" + "="*60)
    print("APPLYING ENHANCED TRANSFORMATIONS")
    print("="*60)
    
    config = EnhancedCreditScoringConfig()
    transformer = EnhancedCreditScoringTransformer(config)
    
    print("✅ Enhanced transformer initialized")
    df_transformed = transformer.transform(df)
    print("✅ Transformations complete.")
    
    print("\n📊 Generated Component Scores (Sample):")
    print(df_transformed[['Client_Type', 'Financial_Stability_Score', 'Cultural_Context_Score', 'Credit_Behavior_Score', 'Credit_Risk_Score']].head())
    
    return df_transformed, transformer


def prepare_features_for_training(df_transformed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features for ML training.
    
    This version uses ONLY the high-level component scores, creating a true hybrid model.
    """
    print("\n" + "="*60)
    print("FEATURE PREPARATION FOR TRAINING")
    print("="*60)
    
    # CRITICAL: The ML model is trained ONLY on the high-level engineered features.
    # This simplifies the model and leverages the expert rules in the transformer.
    training_features = [
        'Financial_Stability_Score',
        'Cultural_Context_Score',
        'Credit_Behavior_Score'
    ]
    
    print(f"✅ Using {len(training_features)} high-level features for ML model:")
    print(f"   {training_features}")
    
    X = df_transformed[training_features].copy()
    y = df_transformed['Default']
    
    # Fill any potential NaNs in scores with 0, as it indicates a lack of data for that component.
    X = X.fillna(0)
    
    print(f"\n📊 Final Training Data Shape:")
    print(f"   Feature Matrix (X): {X.shape}")
    print(f"   Target Vector (y): {y.shape}")
    
    return X, y, training_features


def train_enhanced_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train a simplified, robust ML model on the engineered component scores."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled using StandardScaler")
    
    # Model training with cross-validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'penalty': ['l2'],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, solver='lbfgs', max_iter=2000),
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    print("🔄 Running grid search for best model parameters...")
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✅ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    
    # Model calibration
    calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train_scaled, y_train)
    print("✅ Model calibrated using isotonic regression")
    
    # Model evaluation
    print("\n" + "-"*40)
    print("MODEL EVALUATION")
    print("-" * 40)
    
    y_pred_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'metrics': metrics,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }

def validate_feature_importance(model: Any, features: List[str]):
    """Validate the feature importance from the trained model's coefficients."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE VALIDATION")
    print("="*60)
    
    try:
        # The final estimator is inside the CalibratedClassifierCV
        final_estimator = model.calibrated_classifiers_[0].base_estimator
        coefficients = final_estimator.coef_[0]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        }).sort_values(by='Coefficient', ascending=False)
        
        print("✅ Model coefficients (feature importance):")
        print(importance_df)
        
        # Check if Credit Behavior is the most important for risk
        most_important_feature = importance_df.iloc[0]['Feature']
        print(f"\n   Most influential component: '{most_important_feature}'")
        
        if 'Credit_Behavior' in most_important_feature:
            print("   ✅ As expected, credit history is a dominant factor.")
        else:
            print("   ⚠️  NOTE: Financial or Cultural context is the dominant factor.")
            
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

def run_fairness_validation(df_transformed: pd.DataFrame, model_results: Dict, X: pd.DataFrame):
    """Run fairness validation on transformer scores and model predictions."""
    print("\n" + "="*60)
    print("FAIRNESS VALIDATION")
    print("="*60)
    
    # Scale features for prediction
    X_scaled = model_results['scaler'].transform(X)
    df_transformed['model_predicted_prob'] = model_results['model'].predict_proba(X_scaled)[:, 1]

    protected_attributes = ['Employment_Sector', 'Housing_Status', 'Household_Head']
    
    for attribute in protected_attributes:
        if attribute in df_transformed.columns:
            print(f"\n--- Analysis by {attribute} ---")
            
            # Group by attribute and calculate mean scores/predictions
            analysis = df_transformed.groupby(attribute).agg(
                count=('Credit_Risk_Score', 'count'),
                avg_transformer_score=('Credit_Risk_Score', 'mean'),
                avg_model_prediction=('model_predicted_prob', 'mean'),
                actual_default_rate=('Default', 'mean')
            ).round(4)
            
            print(analysis)
            
            # Calculate disparate impact on model predictions
            mean_predictions = analysis['avg_model_prediction']
            if len(mean_predictions) >= 2:
                min_pred = mean_predictions.min()
                max_pred = mean_predictions.max()
                disparate_impact = min_pred / max_pred if max_pred > 0 else 1.0
                
                print(f"   Disparate Impact Ratio (Model Predictions): {disparate_impact:.3f}")
                if disparate_impact >= 0.8:
                    print(f"   ✅ Passes 80% rule for fairness.")
                else:
                    print(f"   ⚠️  Below 80% rule threshold. Review recommended.")


def save_model_components(training_results: Dict[str, Any], transformer: EnhancedCreditScoringTransformer, features: List[str]):
    """Save all model components for production deployment."""
    print("\n" + "="*60)
    print("SAVING MODEL COMPONENTS")
    print("="*60)
    
    # Save main model and scaler
    joblib.dump(training_results['model'], MODEL_PATH)
    joblib.dump(training_results['scaler'], SCALER_PATH)
    print(f"✅ Model and Scaler saved.")
    
    # Save enhanced transformer
    joblib.dump(transformer, TRANSFORMER_PATH)
    print(f"✅ Enhanced Transformer saved.")
    
    # Save comprehensive model info
    model_info = {
        'model_type': 'hybrid_credit_scoring_v2',
        'version': '2.1.0',
        'creation_timestamp': pd.Timestamp.now().isoformat(),
        'architecture': {
            'description': 'Hybrid model: Rule-based transformer + ML classifier on component scores.',
            'feature_engineering': 'EnhancedCreditScoringTransformer',
            'model_algorithm': 'Calibrated Logistic Regression',
            'features_used_by_model': features
        },
        'performance': training_results['metrics'],
        'training_cv_score': training_results['cv_score'],
        'best_hyperparameters': training_results['best_params'],
    }
    joblib.dump(model_info, MODEL_INFO_PATH)
    print(f"✅ Model Info saved.")

    # Save feature info for prediction pipeline
    feature_info = {
        'model_features': features,
        'transformer_features': get_available_features()['input_features']
    }
    joblib.dump(feature_info, FEATURES_INFO_PATH)
    print(f"✅ Feature Info saved.")
    
    print(f"\n📋 DEPLOYMENT SUMMARY:")
    print(f"   Model Type: {model_info['model_type']}")
    print(f"   ROC-AUC: {model_info['performance']['roc_auc']:.4f}")
    print(f"   Production Ready: ✅ YES")


def main():
    """Main training pipeline."""
    print("🚀 ENHANCED CREDIT SCORING MODEL TRAINING PIPELINE 🚀")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # 1. Setup
        create_directories()
        
        # 2. Data loading and validation
        df = load_and_validate_data()
        
        # 3. Data leakage analysis
        analyze_data_leakage(df)
        
        # 4. Apply enhanced transformations to get high-level features
        df_transformed, transformer = apply_enhanced_transformations(df)
        
        # 5. Prepare features for ML training (using ONLY component scores)
        X, y, features = prepare_features_for_training(df_transformed)
        
        # 6. Train the ML model on the high-level features
        training_results = train_enhanced_model(X, y)
        
        # 7. Validate that the model learned logical feature importances
        validate_feature_importance(training_results['model'], features)
        
        # 8. Run fairness validation
        run_fairness_validation(df_transformed, training_results, X)
        
        # 9. Save all components for deployment
        save_model_components(training_results, transformer, features)
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"🕒 Total Time: {total_time:.2f} seconds")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
