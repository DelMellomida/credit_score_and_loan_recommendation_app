import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

# from fairnessmodel import FairnessAwareModel

# Import SHAP for interpretability analysis
try:
    import shap
    HAS_SHAP = True
    print("SHAP library loaded successfully")
except ImportError:
    HAS_SHAP = False
    print("SHAP library not available. Will skip SHAP analysis.")

# Configuration
RAW_DATA_PATH = '../data/raw/synthetic_training_data13.csv'
MODEL_DIR = '../models'

print("="*80)
print("ROBUST MODEL TRAINING WITH VIF-BASED FEATURE SELECTION")
print("="*80)

# Load and prepare data
print("\n=== LOADING AND PREPROCESSING DATA ===")
df = pd.read_csv(RAW_DATA_PATH)
print(f"Original dataset shape: {df.shape}")

# Check for duplicates
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
if duplicates.sum() > 0:
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

# Filter for new clients
new_clients_df = df[df['Is_Renewing_Client'] == 0].copy()
print(f"New clients dataset shape: {new_clients_df.shape}")

# Check target variable distribution
print(f"\nTarget variable (Default) distribution:")
print(new_clients_df['Default'].value_counts())
print(f"Default rate: {new_clients_df['Default'].mean():.2%}")

# Define initial feature sets
financial_features = [
    'Employment_Sector', 'Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
    'Salary_Frequency', 'Housing_Status', 'Years_at_Current_Address',
    'Number_of_Dependents', 'Comaker_Employment_Tenure_Months',
    'Comaker_Net_Salary_Per_Cutoff', 'Other_Income_Source'
]

cultural_features = [
    'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
    'Paluwagan_Participation', 'Disaster_Preparedness'
]

all_features = financial_features + cultural_features
X = new_clients_df[all_features].copy()
y = new_clients_df['Default']

# Data cleaning and validation
print("\n=== DATA CLEANING ===")

# Numerical columns for outlier handling
numerical_cols = ['Employment_Tenure_Months', 'Net_Salary_Per_Cutoff',
                 'Years_at_Current_Address', 'Number_of_Dependents',
                 'Comaker_Employment_Tenure_Months', 'Comaker_Net_Salary_Per_Cutoff']

# Remove outliers using IQR method
print("Outlier detection and capping for numerical features:")
for col in numerical_cols:
    if col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"  {col}: {outliers} outliers detected and capped")
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

# Ensure non-negative values for numerical features
for col in numerical_cols:
    if col in X.columns and (X[col] < 0).any():
        print(f"  WARNING: Negative values found in {col}, setting to 0")
        X[col] = X[col].clip(lower=0)

# Standardize categorical values
categorical_features = ['Employment_Sector', 'Salary_Frequency', 'Housing_Status', 
                       'Household_Head', 'Comaker_Relationship', 'Has_Community_Role',
                       'Paluwagan_Participation', 'Other_Income_Source', 'Disaster_Preparedness']

print("\nStandardizing categorical values...")
for col in categorical_features:
    if col in X.columns and X[col].dtype == 'object':
        X[col] = X[col].str.strip()
        if col in ['Housing_Status', 'Employment_Sector']:
            X[col] = X[col].str.capitalize()
        elif col in ['Household_Head', 'Has_Community_Role', 'Paluwagan_Participation']:
            X[col] = X[col].str.capitalize()

# Handle missing values
print("\n=== MISSING VALUE IMPUTATION ===")
for col in categorical_features:
    if col in X.columns and X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        if col == 'Comaker_Relationship':
            fill_value = 'Friend'
        elif col == 'Other_Income_Source':
            fill_value = 'None'
        elif col == 'Disaster_Preparedness':
            fill_value = 'None'
        elif col == 'Salary_Frequency':
            fill_value = 'Monthly'
        else:
            mode_value = X[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
        
        X[col].fillna(fill_value, inplace=True)
        print(f"  {col}: Filled {missing_count} missing values with '{fill_value}'")

# Fill numerical missing values with median
for col in numerical_cols:
    if col in X.columns and X[col].isnull().any():
        missing_count = X[col].isnull().sum()
        median_value = X[col].median()
        X[col].fillna(median_value, inplace=True)
        print(f"  {col}: Filled {missing_count} missing values with median {median_value:.2f}")

# Apply capped penalty logic
X['Number_of_Dependents'] = np.minimum(X['Number_of_Dependents'], 5)
print(f"\nApplied capping to Number_of_Dependents (max=5)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train default rate: {y_train.mean():.2%}")
print(f"Test default rate: {y_test.mean():.2%}")

print("\n" + "="*80)
print("PHASE 1: ITERATIVE VIF-BASED FEATURE SELECTION")
print("="*80)

def create_preprocessor_and_vif_check(features_dict, numerical_features):
    """Create preprocessor and calculate VIF for current feature set"""
    
    # Define ordinal and nominal features based on current feature set
    ordinal_features = {}
    nominal_features = []
    
    for feature in features_dict['all_features']:
        if feature == 'Salary_Frequency':
            ordinal_features['Salary_Frequency'] = ['Monthly', 'Bimonthly', 'Biweekly', 'Weekly']
        elif feature == 'Comaker_Relationship':
            ordinal_features['Comaker_Relationship'] = ['Friend', 'Sibling', 'Parent', 'Spouse']
        elif feature == 'Other_Income_Source':
            ordinal_features['Other_Income_Source'] = ['None', 'Freelance', 'Business', 'OFW Remittance']
        elif feature == 'Disaster_Preparedness':
            ordinal_features['Disaster_Preparedness'] = ['None', 'Savings', 'Insurance', 'Community Plan']
        elif feature == 'Paluwagan_Participation':
            ordinal_features['Paluwagan_Participation'] = ['Never', 'Rarely', 'Sometimes', 'Frequently']
        elif feature == 'Has_Community_Role':
            ordinal_features['Has_Community_Role'] = ['None', 'Member', 'Leader', 'Multiple Leader']
        elif feature in categorical_features and feature not in ordinal_features:
            nominal_features.append(feature)
    
    # Create ordinal transformers
    ordinal_transformers = []
    for feature, categories in ordinal_features.items():
        if feature in features_dict['all_features']:
            ordinal_transformers.append(
                (f'ordinal_{feature}',
                 OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1),
                 [feature])
            )
    
    # Filter numerical features to only include those in current feature set
    current_numerical = [f for f in numerical_features if f in features_dict['all_features']]
    current_nominal = [f for f in nominal_features if f in features_dict['all_features']]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), current_numerical),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), current_nominal)
        ] + ordinal_transformers,
        remainder='passthrough'
    )
    
    return preprocessor, ordinal_features, nominal_features

def calculate_vif(X_processed, feature_names):
    """Calculate VIF for all features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_processed, i) for i in range(X_processed.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

# Initialize current feature sets
current_features = {
    'financial_features': financial_features.copy(),
    'cultural_features': cultural_features.copy(),
    'all_features': all_features.copy()
}

print("\n1. Starting iterative VIF-based feature removal...")
iteration = 0
max_iterations = 20  # Safety limit

while iteration < max_iterations:
    iteration += 1
    print(f"\n--- VIF Iteration {iteration} ---")
    print(f"Current feature count: {len(current_features['all_features'])}")
    
    # Create preprocessor for current feature set
    try:
        X_current = X_train[current_features['all_features']].copy()
        preprocessor, ordinal_features, nominal_features = create_preprocessor_and_vif_check(
            current_features, numerical_cols
        )
        
        # Fit preprocessor and transform data
        X_processed = preprocessor.fit_transform(X_current)
        feature_names = preprocessor.get_feature_names_out()
        
        # Calculate VIF
        vif_data = calculate_vif(X_processed, feature_names)
        
        # Find maximum VIF
        max_vif = vif_data.iloc[0]['VIF']
        max_vif_feature = vif_data.iloc[0]['Feature']
        
        print(f"Highest VIF: {max_vif:.2f} for feature '{max_vif_feature}'")
        
        if max_vif <= 5.0:
            print("✓ All features have VIF <= 5.0. VIF cleaning complete!")
            break
        
        # Identify original feature to remove
        # Map processed feature name back to original feature name
        feature_to_remove = None
        for orig_feature in current_features['all_features']:
            if orig_feature in max_vif_feature or max_vif_feature.startswith(f'num__{orig_feature}') or max_vif_feature.startswith(f'nom__{orig_feature}') or max_vif_feature.startswith(f'ordinal_{orig_feature}'):
                feature_to_remove = orig_feature
                break
        
        if feature_to_remove is None:
            # Fallback: try to extract feature name from processed name
            for orig_feature in current_features['all_features']:
                if orig_feature.lower() in max_vif_feature.lower():
                    feature_to_remove = orig_feature
                    break
        
        if feature_to_remove:
            print(f"⚠️  Removing feature: {feature_to_remove}")
            
            # Remove from all relevant lists
            if feature_to_remove in current_features['financial_features']:
                current_features['financial_features'].remove(feature_to_remove)
            if feature_to_remove in current_features['cultural_features']:
                current_features['cultural_features'].remove(feature_to_remove)
            current_features['all_features'].remove(feature_to_remove)
        else:
            print(f"⚠️  Could not identify original feature for '{max_vif_feature}'. Breaking loop.")
            break
            
    except Exception as e:
        print(f"Error in VIF calculation: {str(e)}")
        break

print(f"\n=== VIF CLEANING RESULTS ===")
print(f"Iterations completed: {iteration}")
print(f"Final feature count: {len(current_features['all_features'])}")
print(f"Removed features: {set(all_features) - set(current_features['all_features'])}")
print(f"Remaining features: {current_features['all_features']}")

# Display final VIF scores
if iteration <= max_iterations:
    print(f"\nFinal VIF scores:")
    print(vif_data.head(10).to_string(index=False))

print("\n" + "="*80)
print("PHASE 2: MODEL TRAINING AND COMPARISON")
print("="*80)

# Update X_train and X_test with final feature set
X_train_final = X_train[current_features['all_features']].copy()
X_test_final = X_test[current_features['all_features']].copy()

# Create final preprocessor
final_preprocessor, final_ordinal_features, final_nominal_features = create_preprocessor_and_vif_check(
    current_features, numerical_cols
)

print("\n2. Training Forced-Sensitive Model (C=10.0)...")

# Create forced-sensitive pipeline
forced_sensitive_pipeline = Pipeline([
    ('preprocessor', final_preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        C=10.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    ))
])

# Train forced-sensitive model
forced_sensitive_pipeline.fit(X_train_final, y_train)

# Evaluate forced-sensitive model
y_pred_sensitive = forced_sensitive_pipeline.predict(X_test_final)
y_pred_proba_sensitive = forced_sensitive_pipeline.predict_proba(X_test_final)[:, 1]

print("\nForced-Sensitive Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sensitive):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_sensitive, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_sensitive):.3f}")

print("\n3. Training Final Balanced Model with GridSearchCV...")

# Create balanced pipeline for grid search
balanced_pipeline = Pipeline([
    ('preprocessor', final_preprocessor),
    ('classifier', LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    ))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10]
}

# Perform grid search
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    balanced_pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC score: {grid_search.best_score_:.3f}")

# Get best model
best_balanced_model = grid_search.best_estimator_

print("\n" + "="*80)
print("PHASE 3: FINAL MODEL EVALUATION")
print("="*80)

# Evaluate best balanced model
y_pred_balanced = best_balanced_model.predict(X_test_final)
y_pred_proba_balanced = best_balanced_model.predict_proba(X_test_final)[:, 1]

print("\n4. Final Balanced Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred_balanced, zero_division=0):.3f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.3f}")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_balanced)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Final Balanced Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_balanced)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba_balanced):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Balanced Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n5. SHAP Analysis...")

if HAS_SHAP:
    try:
        # Get processed data for SHAP
        X_train_processed = forced_sensitive_pipeline.named_steps['preprocessor'].transform(X_train_final)
        X_test_processed = best_balanced_model.named_steps['preprocessor'].transform(X_test_final)
        processed_feature_names = best_balanced_model.named_steps['preprocessor'].get_feature_names_out()
        
        print("Calculating SHAP values for both models...")
        
        # SHAP for forced-sensitive model
        explainer_sensitive = shap.LinearExplainer(
            forced_sensitive_pipeline.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        shap_values_sensitive = explainer_sensitive.shap_values(X_test_processed)
        
        # SHAP for balanced model
        explainer_balanced = shap.LinearExplainer(
            best_balanced_model.named_steps['classifier'],
            X_train_processed,
            feature_perturbation="interventional"
        )
        shap_values_balanced = explainer_balanced.shap_values(X_test_processed)
        
        # Create stacked SHAP comparison plots for better readability
        print("Generating stacked SHAP comparison plots...")

        # Create figure with vertical stacking for better feature name readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 28))

        # Top subplot - Sensitive model
        plt.sca(ax1)
        shap.summary_plot(shap_values_sensitive, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        ax1.set_title('Forced-Sensitive Model (C=10.0)', fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        # Bottom subplot - Balanced model
        plt.sca(ax2)
        shap.summary_plot(shap_values_balanced, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        ax2.set_title(f"Final Balanced Model (C={best_balanced_model.named_steps['classifier'].C})", 
                    fontsize=18, fontweight='bold', pad=20)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.tick_params(axis='x', labelsize=12)

        # Adjust spacing between plots
        plt.tight_layout(pad=4.0)
        plt.show()

        # Alternative: Create individual plots if stacked version is still too cramped
        print("\nAlternative: Creating individual SHAP plots...")

        # Individual plot for Sensitive model
        plt.figure(figsize=(16, 12))
        shap.summary_plot(shap_values_sensitive, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        plt.title('Forced-Sensitive Model (C=10.0)', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

        # Individual plot for Balanced model
        plt.figure(figsize=(16, 12))
        shap.summary_plot(shap_values_balanced, X_test_processed, 
                        feature_names=processed_feature_names, 
                        show=False, 
                        max_display=len(processed_feature_names))
        plt.title(f"Final Balanced Model (C={best_balanced_model.named_steps['classifier'].C})", 
                fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        print("Continuing without SHAP visualization...")

print("\n" + "="*80)
print("PHASE 4: ALGORITHMIC FAIRNESS AND BIAS ASSESSMENT")
print("="*80)

# Import the necessary library
try:
    from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
    from fairlearn.postprocessing import ThresholdOptimizer
    HAS_FAIRLEARN = True
    print("Fairlearn library loaded successfully.")
except ImportError:
    HAS_FAIRLEARN = False
    print("Fairlearn library not available. Skipping bias assessment.")
    print("To install: pip install fairlearn")

if HAS_FAIRLEARN:
    # Define sensitive features to audit
    sensitive_features_to_audit = {
        'Paluwagan_Participation': 'To assess if the model unfairly favors individuals who participate in informal savings groups',
        'Has_Community_Role': 'To check for bias between individuals with formal community roles versus those without',
        'Household_Head': 'To ensure fairness regardless of the applicant\'s role within their household'
    }
    
    # Store results for all sensitive features
    fairness_results = {}
    
    for sensitive_feature_name, description in sensitive_features_to_audit.items():
        # Check if the feature is still in the model after VIF removal
        if sensitive_feature_name not in X_test_final.columns:
            print(f"\n⚠️  Sensitive feature '{sensitive_feature_name}' was removed during VIF cleaning. Skipping fairness audit for this feature.")
            continue
            
        print(f"\n{'='*60}")
        print(f"Conducting fairness audit for: '{sensitive_feature_name}'")
        print(f"Purpose: {description}")
        print(f"{'='*60}")
        
        sensitive_feature_series = X_test_final[sensitive_feature_name]
        
        # Check distribution of sensitive feature
        print(f"\nDistribution of '{sensitive_feature_name}' in test set:")
        print(sensitive_feature_series.value_counts())
        
        # Group metrics by the sensitive feature
        grouped_metrics = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'precision': precision_score,
                'recall': recall_score,
                'selection_rate': lambda y_true, y_pred: y_pred.mean()
            },
            y_true=y_test,
            y_pred=y_pred_balanced,
            sensitive_features=sensitive_feature_series
        )
        
        print(f"\n--- Metrics by Group (Before Mitigation) ---")
        print(grouped_metrics.by_group)
        
        # Calculate fairness metrics
        dpd = demographic_parity_difference(y_test, y_pred_balanced, sensitive_features=sensitive_feature_series)
        eod = equalized_odds_difference(y_test, y_pred_balanced, sensitive_features=sensitive_feature_series)
        
        print(f"\nDemographic Parity Difference: {dpd:.4f}")
        print(f"Equalized Odds Difference: {eod:.4f}")
        print("(A value closer to 0 is fairer)")
        
        # Check if mitigation is needed (threshold = 0.05)
        fairness_threshold = 0.05
        needs_mitigation = abs(dpd) > fairness_threshold or abs(eod) > fairness_threshold
        
        if needs_mitigation:
            print(f"\n⚠️  Bias detected (exceeds threshold of {fairness_threshold}). Applying mitigation...")
            
            # Apply ThresholdOptimizer for mitigation
            print(f"\n--- Applying Bias Mitigation using ThresholdOptimizer ---")
            
            # Create and fit the postprocessing optimizer
            postprocess_est = ThresholdOptimizer(
                estimator=best_balanced_model,
                constraints="equalized_odds",
                objective="balanced_accuracy_score",
                prefit=True,
                predict_method="predict_proba"
            )
            
            # Fit on training data to learn optimal thresholds
            postprocess_est.fit(
                X_train_final, 
                y_train, 
                sensitive_features=X_train_final[sensitive_feature_name]
            )
            
            # Get mitigated predictions
            y_pred_mitigated = postprocess_est.predict(
                X_test_final, 
                sensitive_features=sensitive_feature_series
            )
            
            # Recalculate fairness metrics after mitigation
            print(f"\n--- Fairness Metrics (After Mitigation) ---")
            dpd_mitigated = demographic_parity_difference(
                y_test, y_pred_mitigated, sensitive_features=sensitive_feature_series
            )
            eod_mitigated = equalized_odds_difference(
                y_test, y_pred_mitigated, sensitive_features=sensitive_feature_series
            )
            
            print(f"Mitigated Demographic Parity Difference: {dpd_mitigated:.4f} (was {dpd:.4f})")
            print(f"Mitigated Equalized Odds Difference: {eod_mitigated:.4f} (was {eod:.4f})")
            
            # Performance trade-off analysis
            print(f"\n--- Performance Trade-off Analysis ---")
            print(f"{'Metric':<20} {'Original':<12} {'Mitigated':<12} {'Change':<12}")
            print("-" * 56)
            
            # Calculate performance metrics for mitigated model
            acc_original = accuracy_score(y_test, y_pred_balanced)
            acc_mitigated = accuracy_score(y_test, y_pred_mitigated)
            prec_original = precision_score(y_test, y_pred_balanced, zero_division=0)
            prec_mitigated = precision_score(y_test, y_pred_mitigated, zero_division=0)
            rec_original = recall_score(y_test, y_pred_balanced, zero_division=0)
            rec_mitigated = recall_score(y_test, y_pred_mitigated, zero_division=0)
            f1_original = f1_score(y_test, y_pred_balanced, zero_division=0)
            f1_mitigated = f1_score(y_test, y_pred_mitigated, zero_division=0)
            
            print(f"{'Accuracy':<20} {acc_original:<12.3f} {acc_mitigated:<12.3f} {acc_mitigated - acc_original:+12.3f}")
            print(f"{'Precision':<20} {prec_original:<12.3f} {prec_mitigated:<12.3f} {prec_mitigated - prec_original:+12.3f}")
            print(f"{'Recall':<20} {rec_original:<12.3f} {rec_mitigated:<12.3f} {rec_mitigated - rec_original:+12.3f}")
            print(f"{'F1 Score':<20} {f1_original:<12.3f} {f1_mitigated:<12.3f} {f1_mitigated - f1_original:+12.3f}")
            
            # Store results
            fairness_results[sensitive_feature_name] = {
                'needs_mitigation': True,
                'postprocessor': postprocess_est,
                'dpd_original': dpd,
                'dpd_mitigated': dpd_mitigated,
                'eod_original': eod,
                'eod_mitigated': eod_mitigated,
                'performance_change': {
                    'accuracy': acc_mitigated - acc_original,
                    'precision': prec_mitigated - prec_original,
                    'recall': rec_mitigated - rec_original,
                    'f1_score': f1_mitigated - f1_original
                }
            }
            
            # Visualization of group-wise performance
            print(f"\n--- Creating Fairness Visualization ---")
            
            # Get group-wise metrics for visualization
            grouped_mitigated = MetricFrame(
                metrics={
                    'selection_rate': lambda y_true, y_pred: y_pred.mean(),
                    'accuracy': accuracy_score
                },
                y_true=y_test,
                y_pred=y_pred_mitigated,
                sensitive_features=sensitive_feature_series
            )
            
            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Selection rates comparison
            groups = grouped_metrics.by_group.index
            x_pos = np.arange(len(groups))
            
            ax1.bar(x_pos - 0.2, grouped_metrics.by_group['selection_rate'], 
                   0.4, label='Original', alpha=0.7)
            ax1.bar(x_pos + 0.2, grouped_mitigated.by_group['selection_rate'], 
                   0.4, label='Mitigated', alpha=0.7)
            ax1.set_xlabel(sensitive_feature_name)
            ax1.set_ylabel('Selection Rate (Approval Rate)')
            ax1.set_title(f'Selection Rates by {sensitive_feature_name}')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(groups, rotation=45 if len(groups) > 3 else 0)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy comparison
            ax2.bar(x_pos - 0.2, grouped_metrics.by_group['accuracy'], 
                   0.4, label='Original', alpha=0.7)
            ax2.bar(x_pos + 0.2, grouped_mitigated.by_group['accuracy'], 
                   0.4, label='Mitigated', alpha=0.7)
            ax2.set_xlabel(sensitive_feature_name)
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'Accuracy by {sensitive_feature_name}')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(groups, rotation=45 if len(groups) > 3 else 0)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        else:
            print(f"\n✅ No significant bias detected for '{sensitive_feature_name}' (within threshold of {fairness_threshold})")
            fairness_results[sensitive_feature_name] = {
                'needs_mitigation': False,
                'dpd_original': dpd,
                'eod_original': eod
            }
    
    # Summary of fairness audit
    print(f"\n{'='*80}")
    print("FAIRNESS AUDIT SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nSensitive features audited: {len(fairness_results)}")
    features_needing_mitigation = [f for f, r in fairness_results.items() if r['needs_mitigation']]
    
    if features_needing_mitigation:
        print(f"\n⚠️  Features requiring mitigation: {features_needing_mitigation}")
        
        # Save the fairness-aware model with postprocessors
        print(f"\n--- Creating Fairness-Aware Model Wrapper ---")
        
        # Create a wrapper class for the fair model
        class FairnessAwareModel:
            def __init__(self, base_model, fairness_postprocessors):
                self.base_model = base_model
                self.fairness_postprocessors = fairness_postprocessors
                
            def predict(self, X, apply_fairness=True, sensitive_feature_name=None):
                if not apply_fairness or not sensitive_feature_name:
                    return self.base_model.predict(X)
                
                if sensitive_feature_name in self.fairness_postprocessors:
                    postprocessor = self.fairness_postprocessors[sensitive_feature_name]
                    return postprocessor.predict(
                        X, 
                        sensitive_features=X[sensitive_feature_name]
                    )
                else:
                    return self.base_model.predict(X)
                    
            def predict_proba(self, X):
                return self.base_model.predict_proba(X)
        
        # Create fairness-aware model
        fairness_postprocessors = {
            feature: results['postprocessor'] 
            for feature, results in fairness_results.items() 
            if results['needs_mitigation']
        }
        
        fairness_aware_model = FairnessAwareModel(
            best_balanced_model, 
            fairness_postprocessors
        )
        
        print("✅ Fairness-aware model wrapper created successfully")
        
        # Update the model to be saved
        model_to_save = fairness_aware_model
        model_filename = "new_client_model_fair.pkl"
        
    else:
        print("\n✅ No features required mitigation. The model is already fair!")
        model_to_save = best_balanced_model
        model_filename = "new_client_model.pkl"
    
    # Save fairness audit report
    print(f"\n--- Saving Fairness Audit Report ---")
    
    fairness_report = {
        'audit_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sensitive_features_audited': list(fairness_results.keys()),
        'fairness_threshold': fairness_threshold,
        'results': {}
    }
    
    for feature, results in fairness_results.items():
        fairness_report['results'][feature] = {
            'needs_mitigation': results['needs_mitigation'],
            'demographic_parity_difference': {
                'original': results['dpd_original'],
                'mitigated': results.get('dpd_mitigated', None)
            },
            'equalized_odds_difference': {
                'original': results['eod_original'],
                'mitigated': results.get('eod_mitigated', None)
            },
            'performance_impact': results.get('performance_change', None)
        }
    
    # Save report as JSON
    report_path = f"{MODEL_DIR}/fairness_audit_report.json"
    with open(report_path, 'w') as f:
        json.dump(fairness_report, f, indent=2)
    
    print(f"✓ Fairness audit report saved to: {report_path}")
    
else:
    print("\n⚠️  Fairlearn not installed. Skipping fairness audit.")
    print("To enable fairness auditing, install fairlearn: pip install fairlearn")
    model_to_save = best_balanced_model
    model_filename = "new_client_model.pkl"

print("\n6. Serializing Final Model...")

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Determine which model to save based on fairness audit results
if 'model_to_save' not in locals():
    model_to_save = best_balanced_model
    model_filename = "new_client_model.pkl"

if 'model_filename' not in locals():
    model_filename = "new_client_model.pkl"

# Save the model
model_path = f"{MODEL_DIR}/{model_filename}"
with open(model_path, 'wb') as f:
    pickle.dump(model_to_save, f)

print(f"✓ Model saved successfully to: {model_path}")

# Verify saved model
print("\nVerifying saved model...")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Test predictions based on model type
if hasattr(loaded_model, 'fairness_postprocessors'):
    print("✓ Loaded fairness-aware model with postprocessors")
    # Test without fairness correction
    sample_predictions_base = loaded_model.predict(X_test_final[:5], apply_fairness=False)
    print(f"Sample predictions (base model): {sample_predictions_base}")
    
    # Test with fairness correction if applicable
    for sensitive_feature in loaded_model.fairness_postprocessors.keys():
        if sensitive_feature in X_test_final.columns:
            sample_predictions_fair = loaded_model.predict(
                X_test_final[:5], 
                apply_fairness=True, 
                sensitive_feature_name=sensitive_feature
            )
            print(f"Sample predictions (fair for {sensitive_feature}): {sample_predictions_fair}")
            break
else:
    sample_predictions = loaded_model.predict(X_test_final[:5])
    print(f"Sample predictions from loaded model: {sample_predictions}")

print("✓ Model pipeline successfully serialized and verified!")

# Save model metadata
metadata = {
    'model_type': 'FairnessAwareModel' if hasattr(model_to_save, 'fairness_postprocessors') else 'StandardModel',
    'features_used': current_features['all_features'],
    'features_removed_vif': list(set(all_features) - set(current_features['all_features'])),
    'regularization_parameter': best_balanced_model.named_steps['classifier'].C,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_roc_auc': roc_auc_score(y_test, y_pred_proba_balanced),
    'fairness_applied': hasattr(model_to_save, 'fairness_postprocessors')
}

metadata_path = f"{MODEL_DIR}/model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model metadata saved to: {metadata_path}")

print("\n" + "="*80)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*80)

print(f"\n📊 FINAL RESULTS:")
print(f"• Features used: {len(current_features['all_features'])}")
print(f"• Features removed due to multicollinearity: {len(all_features) - len(current_features['all_features'])}")
print(f"• Optimal regularization parameter: C = {best_balanced_model.named_steps['classifier'].C}")
print(f"• Final model ROC AUC: {roc_auc_score(y_test, y_pred_proba_balanced):.3f}")

print(f"\n🔧 MODEL CONFIGURATION:")
print(f"• Algorithm: Logistic Regression with L2 regularization")
print(f"• Class balancing: Balanced class weights")
print(f"• Cross-validation: 3-fold stratified")
print(f"• VIF threshold: ≤ 5.0 (multicollinearity removed)")

if HAS_FAIRLEARN and features_needing_mitigation:
    print(f"\n⚖️  FAIRNESS CONFIGURATION:")
    print(f"• Fairness threshold: {fairness_threshold}")
    print(f"• Mitigation applied for: {features_needing_mitigation}")
    print(f"• Mitigation method: ThresholdOptimizer with equalized odds constraint")

print(f"\n📁 DELIVERABLES:")
print(f"• Model file: {model_path}")
print(f"• Model metadata: {metadata_path}")
if HAS_FAIRLEARN:
    print(f"• Fairness audit report: {report_path}")
print(f"• Pipeline includes: Preprocessor + Trained Classifier")
if hasattr(model_to_save, 'fairness_postprocessors'):
    print(f"• Fairness-aware wrapper with bias mitigation")
print(f"• Ready for production deployment")

# Performance comparison
print(f"\n📈 MODEL COMPARISON:")
print(f"{'Metric':<15} {'Sensitive':<12} {'Balanced':<12} {'Difference':<12}")
print("-" * 55)
print(f"{'ROC AUC':<15} {roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced):<12.3f} {roc_auc_score(y_test, y_pred_proba_balanced) - roc_auc_score(y_test, y_pred_proba_sensitive):<12.3f}")
print(f"{'Precision':<15} {precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {precision_score(y_test, y_pred_balanced, zero_division=0) - precision_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")
print(f"{'Recall':<15} {recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0):<12.3f} {recall_score(y_test, y_pred_balanced, zero_division=0) - recall_score(y_test, y_pred_sensitive, zero_division=0):<12.3f}")

print(f"\n✅ TRAINING COMPLETE!")
print("The final model has been successfully trained, validated, and serialized.")
if hasattr(model_to_save, 'fairness_postprocessors'):
    print("The model includes fairness-aware post-processing to ensure equitable treatment across sensitive groups.")