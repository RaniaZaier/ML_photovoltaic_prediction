"""
Model Comparison Script: XGBoost vs Linear Regression vs Random Forest vs KNN
==============================================================================
This script trains and compares 4 different machine learning algorithms
on your organic photovoltaic dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import warnings
import pickle

warnings.filterwarnings("ignore")

print("=" * 70)
print("MODEL COMPARISON: XGBoost vs Linear Regression vs Random Forest vs KNN")
print("=" * 70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n📂 Loading data...")
data = pd.read_csv('data_aromatic_converted.csv', encoding='latin1', sep=';')

# Fix BOM and convert European numbers
data.columns = [col.replace('ï»¿', '') for col in data.columns]

numeric_columns = ['D_HOMO', 'D_LUMO', 'D_lamda_max', 'A_HOMO', 'A_LUMO',
                   'A_lamda_max', 'VOC (V)', 'JSC (mA/cmÂ²)', 'FF (%)', 'PCE (%)']

for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

if data['FF (%)'].max() < 10:
    data['FF (%)'] = data['FF (%)'] * 100


# Calculate molecular descriptors
def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 7
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.MolLogP(mol),
        Descriptors.RingCount(mol)
    ]


desc_names = ['MolWt', 'TPSA', 'MolMR', 'NumHAcceptors', 'NumRotatableBonds', 'MolLogP', 'RingCount']
desc_df = data['D_SMILES'].apply(mol_descriptors).apply(pd.Series)
desc_df.columns = desc_names

data_with_descriptors = pd.concat([data, desc_df], axis=1).dropna()

# Feature engineering
data_with_descriptors['D_Bandgap'] = data_with_descriptors['D_LUMO'] - data_with_descriptors['D_HOMO']
data_with_descriptors['A_Bandgap'] = data_with_descriptors['A_LUMO'] - data_with_descriptors['A_HOMO']
data_with_descriptors['HOMO_offset'] = data_with_descriptors['D_HOMO'] - data_with_descriptors['A_HOMO']
data_with_descriptors['LUMO_offset'] = data_with_descriptors['D_LUMO'] - data_with_descriptors['A_LUMO']
data_with_descriptors['Lambda_gap'] = data_with_descriptors['A_lamda_max'] - data_with_descriptors['D_lamda_max']

data_with_descriptors = data_with_descriptors.drop(columns=['MolMR'])

# Prepare features and targets
target_cols = ['PCE (%)', 'FF (%)', 'JSC (mA/cmÂ²)', 'VOC (V)']
exclude_cols = ['D_Name', 'D_SMILES', 'Solvent', 'A_Name', 'REFERENCE', 'Conversion_Status'] + target_cols
feature_cols = [col for col in data_with_descriptors.columns
                if col not in exclude_cols and
                data_with_descriptors[col].dtype in ['int64', 'float64']]

X = data_with_descriptors[feature_cols].copy()
y = data_with_descriptors[target_cols].copy()

print(f"✅ Dataset prepared: {len(X)} samples, {len(feature_cols)} features")

# ============================================================================
# 2. DEFINE MODELS AND HYPERPARAMETERS
# ============================================================================
print("\n🤖 Defining models...")

# Model configurations with hyperparameter grids
model_configs = {
    'XGBoost': {
        'model': XGBRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0]
        },
        'color': '#FF6B6B'
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'color': '#4ECDC4'
    },
    'Linear Regression': {
        'model': Ridge(random_state=42),  # Ridge for regularization
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0]
        },
        'color': '#95E1D3'
    },
    'KNN': {
        'model': KNeighborsRegressor(n_jobs=-1),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'color': '#F38181'
    }
}

# ============================================================================
# 3. TRAIN AND EVALUATE ALL MODELS
# ============================================================================
print("\n🔧 Training and evaluating models (this will take a few minutes)...")
print("=" * 70)

cv = KFold(n_splits=10, shuffle=True, random_state=42)
results = {}

# Scale features (important for KNN and Linear Regression)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

for model_name, config in model_configs.items():
    print(f"\n🎯 Training {model_name}...")
    results[model_name] = {}

    # Use scaled data for KNN and Linear Regression
    X_train = X_scaled if model_name in ['KNN', 'Linear Regression'] else X

    for target in target_cols:
        print(f"  ⏳ {target}...", end=" ")

        # Hyperparameter tuning with GridSearchCV
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y[target])
        best_model = grid_search.best_estimator_

        # Cross-validation evaluation
        r2_scores = cross_val_score(best_model, X_train, y[target], cv=cv, scoring='r2')
        neg_mse = cross_val_score(best_model, X_train, y[target], cv=cv, scoring='neg_mean_squared_error')
        neg_mae = cross_val_score(best_model, X_train, y[target], cv=cv, scoring='neg_mean_absolute_error')

        rmse_scores = np.sqrt(-neg_mse)
        mae_scores = -neg_mae

        results[model_name][target] = {
            'best_model': best_model,
            'best_params': grid_search.best_params_,
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }

        print(f"R² = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

# ============================================================================
# 4. COMPARISON SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON SUMMARY: 10-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

# Create comparison dataframe
comparison_data = []
for model_name in model_configs.keys():
    for target in target_cols:
        comparison_data.append({
            'Model': model_name,
            'Target': target,
            'R²': results[model_name][target]['r2_mean'],
            'R²_std': results[model_name][target]['r2_std'],
            'RMSE': results[model_name][target]['rmse_mean'],
            'RMSE_std': results[model_name][target]['rmse_std'],
            'MAE': results[model_name][target]['mae_mean']
        })

comparison_df = pd.DataFrame(comparison_data)

# Print results by target
for target in target_cols:
    print(f"\n{target}:")
    target_results = comparison_df[comparison_df['Target'] == target].copy()
    target_results = target_results.sort_values('R²', ascending=False)

    print(f"{'Model':<20} {'R²':<15} {'RMSE':<15} {'MAE':<15}")
    print("-" * 70)
    for _, row in target_results.iterrows():
        print(f"{row['Model']:<20} {row['R²']:.3f} ± {row['R²_std']:.3f}   "
              f"{row['RMSE']:.3f} ± {row['RMSE_std']:.3f}   {row['MAE']:.3f}")

# ============================================================================
# 5. VISUALIZATION: MODEL COMPARISON
# ============================================================================
print("\n📊 Generating comparison visualizations...")

# 5.1: R² Comparison by Target
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, target in enumerate(target_cols):
    ax = axes[idx]
    target_data = comparison_df[comparison_df['Target'] == target].sort_values('R²', ascending=True)

    colors = [model_configs[model]['color'] for model in target_data['Model']]
    bars = ax.barh(range(len(target_data)), target_data['R²'],
                   xerr=target_data['R²_std'], color=colors, alpha=0.8, capsize=5)

    ax.set_yticks(range(len(target_data)))
    ax.set_yticklabels(target_data['Model'], fontsize=11)
    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{target}', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()

    # Add value labels
    for i, (r2, model) in enumerate(zip(target_data['R²'], target_data['Model'])):
        ax.text(r2 + 0.02, i, f'{r2:.3f}', va='center', fontsize=9)

plt.suptitle('Model Comparison: R² Scores by Target (10-Fold CV)',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('model_comparison_r2.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison_r2.png")

# 5.2: Overall Performance Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
pivot_r2 = comparison_df.pivot(index='Model', columns='Target', values='R²')
sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1.0,
            cbar_kws={'label': 'R² Score'}, ax=ax, linewidths=1)
ax.set_title('Model Performance Heatmap (R² Scores)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Target Property', fontsize=12, fontweight='bold')
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison_heatmap.png")

# 5.3: RMSE Comparison
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(target_cols))
width = 0.2

for idx, model_name in enumerate(model_configs.keys()):
    rmse_values = [results[model_name][target]['rmse_mean'] for target in target_cols]
    rmse_stds = [results[model_name][target]['rmse_std'] for target in target_cols]
    offset = (idx - 1.5) * width
    ax.bar(x + offset, rmse_values, width, yerr=rmse_stds,
           label=model_name, color=model_configs[model_name]['color'],
           alpha=0.8, capsize=3)

ax.set_xlabel('Target Property', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: RMSE by Target (Lower is Better)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(target_cols, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_rmse.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison_rmse.png")

# ============================================================================
# 6. BEST MODEL SELECTION
# ============================================================================
print("\n" + "=" * 70)
print("BEST MODEL FOR EACH TARGET")
print("=" * 70)

best_models = {}
for target in target_cols:
    target_results = comparison_df[comparison_df['Target'] == target]
    best_idx = target_results['R²'].idxmax()
    best_model_name = target_results.loc[best_idx, 'Model']
    best_r2 = target_results.loc[best_idx, 'R²']

    print(f"\n{target}:")
    print(f"  🏆 Best Model: {best_model_name}")
    print(f"  📊 R² Score: {best_r2:.3f}")

    # Store best model
    X_train = X_scaled if best_model_name in ['KNN', 'Linear Regression'] else X
    best_models[target] = {
        'model_name': best_model_name,
        'model': results[best_model_name][target]['best_model'],
        'scaler': scaler if best_model_name in ['KNN', 'Linear Regression'] else None
    }

    # Train on full dataset
    best_models[target]['model'].fit(X_train, y[target])

# ============================================================================
# 7. SAVE BEST MODELS
# ============================================================================
print("\n💾 Saving best models...")

save_data = {
    'best_models': best_models,
    'feature_cols': feature_cols,
    'all_results': results,
    'comparison_df': comparison_df,
    'model_configs': {k: v for k, v in model_configs.items() if k != 'model'}  # Remove model objects
}

with open('opv_models_comparison.pkl', 'wb') as f:
    pickle.dump(save_data, f)

print("✅ Saved: opv_models_comparison.pkl")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Count wins per model
wins = {}
for model in model_configs.keys():
    wins[model] = 0

for target in target_cols:
    target_results = comparison_df[comparison_df['Target'] == target]
    best_model = target_results.loc[target_results['R²'].idxmax(), 'Model']
    wins[best_model] += 1

print("\n🏆 Overall Performance (Wins by Target):")
for model, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
    print(f"  {model}: {count}/4 targets")

# Average R² across all targets
print("\n📊 Average R² Across All Targets:")
for model in model_configs.keys():
    avg_r2 = comparison_df[comparison_df['Model'] == model]['R²'].mean()
    print(f"  {model}: {avg_r2:.3f}")

print("\n" + "=" * 70)
print("✅ COMPARISON COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  📊 model_comparison_r2.png")
print("  📊 model_comparison_heatmap.png")
print("  📊 model_comparison_rmse.png")
print("  💾 opv_models_comparison.pkl")
print("\n" + "=" * 70)