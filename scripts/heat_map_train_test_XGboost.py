import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore")

print("=" * 80)
print("REALISTIC ML MODEL FOR SMALL OPV DATASET (63 samples)")
print("=" * 80)
print("\n⚠️  IMPORTANT: With 63 samples, expect limited predictive power")
print("   This is a DATA LIMITATION, not an algorithm problem")
print("   Models will have high uncertainty - use with caution!")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
data = pd.read_csv('data_cleande_aromatic.csv', encoding='latin1', sep=';', index_col=0)
#data.columns = [col.replace('ï»¿', '') for col in data.columns]

# Convert European format
numeric_columns = ['D_HOMO', 'D_LUMO', 'D_lamda_max', 'A_HOMO', 'A_LUMO',
                   'A_lamda_max', 'VOC (V)', 'JSC (mA/cmÂ²)', 'FF (%)', 'PCE (%)']

for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Fix FF outliers (if any remain)
if (data['FF (%)'] < 10).any():
    data.loc[data['FF (%)'] < 1, 'FF (%)'] *= 100
    data.loc[(data['FF (%)'] >= 1) & (data['FF (%)'] < 10), 'FF (%)'] *= 10


# Calculate descriptors
def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 6
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.MolLogP(mol),
        Descriptors.RingCount(mol)
    ]


desc_names = ['MolWt', 'TPSA', 'NumHAcceptors', 'NumRotatableBonds', 'MolLogP', 'RingCount']
desc_df = data['D_SMILES'].apply(mol_descriptors).apply(pd.Series)
desc_df.columns = desc_names

data_full = pd.concat([data, desc_df], axis=1)
data_full = data_full.dropna()

# Feature engineering
data_full['D_Bandgap'] = data_full['D_LUMO'] - data_full['D_HOMO']
data_full['A_Bandgap'] = data_full['A_LUMO'] - data_full['A_HOMO']
data_full['HOMO_offset'] = data_full['D_HOMO'] - data_full['A_HOMO']
data_full['LUMO_offset'] = data_full['D_LUMO'] - data_full['A_LUMO']

print(f"\n✅ Dataset prepared: {len(data_full)} samples")

# ============================================================================
# 2. STRATEGY: USE FEWER FEATURES (5 per target based on correlation)
# ============================================================================
target_cols = ['PCE (%)', 'FF (%)', 'JSC (mA/cmÂ²)', 'VOC (V)']

all_features = ['D_HOMO', 'D_LUMO', 'D_lamda_max', 'A_HOMO', 'A_LUMO', 'A_lamda_max',
                'MolWt', 'TPSA', 'NumHAcceptors', 'NumRotatableBonds', 'MolLogP',
                'RingCount', 'D_Bandgap', 'A_Bandgap', 'HOMO_offset', 'LUMO_offset']

X_all = data_full[all_features]
y = data_full[target_cols]

print("\n" + "=" * 80)
print("FEATURE SELECTION: Top 6 features per target (reduce overfitting)")
print("=" * 80)

selected_features = {}
for target in target_cols:
    correlations = X_all.corrwith(y[target]).abs().sort_values(ascending=False)
    top_6 = correlations.head(6).index.tolist()
    selected_features[target] = top_6

    print(f"\n{target}:")
    for feat in top_6:
        print(f"   {feat:20s}: r = {correlations[feat]:5.3f}")

# ============================================================================
# 3. BUILD MODELS WITH REALISTIC EXPECTATIONS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING (5-Fold CV for stability)")
print("=" * 80)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
final_models = {}
final_scalers = {}
cv_results = {}

for target in target_cols:
    print(f"\n{'=' * 80}")
    print(f"TARGET: {target}")
    print(f"{'=' * 80}")

    X_target = X_all[selected_features[target]]

    # Test multiple models
    models_to_test = {
        'Ridge (α=5)': Ridge(alpha=5.0),
        'XGBoost (conservative)': XGBRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=3,
            random_state=42
        )
    }

    best_model = None
    best_model_name = None
    best_score = -np.inf
    best_scaler = None

    for model_name, model in models_to_test.items():
        # Ridge needs scaling, XGBoost doesn't
        if 'Ridge' in model_name:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_target)
            scores = cross_val_score(model, X_scaled, y[target], cv=cv, scoring='r2')
        else:
            scaler = None
            scores = cross_val_score(model, X_target, y[target], cv=cv, scoring='r2')

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"\n{model_name}:")
        print(f"   CV R²: {mean_score:.3f} ± {std_score:.3f}")
        print(f"   Individual folds: {scores}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = model_name
            best_scaler = scaler

    print(f"\n✅ SELECTED: {best_model_name} (R² = {best_score:.3f})")

    # Train final model on full data
    if best_scaler is not None:
        X_final = best_scaler.fit_transform(X_target)
    else:
        X_final = X_target

    best_model.fit(X_final, y[target])

    # Calculate metrics
    y_pred_train = best_model.predict(X_final)
    train_r2 = r2_score(y[target], y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y[target], y_pred_train))

    # Get CV predictions for visualization
    y_pred_cv = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X_target):
        if best_scaler is not None:
            X_train = best_scaler.fit_transform(X_target.iloc[train_idx])
            X_test = best_scaler.transform(X_target.iloc[test_idx])
        else:
            X_train = X_target.iloc[train_idx]
            X_test = X_target.iloc[test_idx]

        best_model.fit(X_train, y[target].iloc[train_idx])
        y_pred_cv[test_idx] = best_model.predict(X_test)

    cv_r2 = r2_score(y[target], y_pred_cv)
    cv_rmse = np.sqrt(mean_squared_error(y[target], y_pred_cv))

    print(f"   Training R²: {train_r2:.3f} (RMSE: {train_rmse:.3f})")
    print(f"   CV R²: {cv_r2:.3f} (RMSE: {cv_rmse:.3f})")

    # Interpret results
    if cv_r2 < 0:
        print(f"   ⚠️  Model worse than baseline - predictions unreliable")
    elif cv_r2 < 0.3:
        print(f"   ⚠️  Weak predictive power - use with extreme caution")
    elif cv_r2 < 0.6:
        print(f"   ⚙️  Moderate predictive power - reasonable for screening")
    else:
        print(f"   ✅ Good predictive power for this dataset size")

    final_models[target] = best_model
    final_scalers[target] = best_scaler
    cv_results[target] = {
        'model_name': best_model_name,
        'cv_r2': cv_r2,
        'cv_rmse': cv_rmse,
        'train_r2': train_r2,
        'y_pred_cv': y_pred_cv
    }

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: CV predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, target in enumerate(target_cols):
    ax = axes[i]

    y_true = y[target]
    y_pred = cv_results[target]['y_pred_cv']
    cv_r2 = cv_results[target]['cv_r2']
    cv_rmse = cv_results[target]['cv_rmse']
    model_name = cv_results[target]['model_name']

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=80, edgecolors='black', linewidth=0.8)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

    # Styling
    ax.set_xlabel(f'True {target}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target}', fontsize=12, fontweight='bold')
    ax.set_title(f'{target}\n{model_name}\nCV R² = {cv_r2:.3f}, RMSE = {cv_rmse:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Add uncertainty bands
    residuals = y_true - y_pred
    std_residual = np.std(residuals)
    ax.fill_between([min_val, max_val],
                    [min_val - std_residual, max_val - std_residual],
                    [min_val + std_residual, max_val + std_residual],
                    alpha=0.2, color='gray', label=f'±1σ ({std_residual:.2f})')

plt.suptitle('Cross-Validation Predictions (5-Fold, 6 features per target)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('realistic_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: realistic_predictions.png")

# Plot 2: Model comparison summary
fig, ax = plt.subplots(figsize=(10, 6))

targets_plot = [t.replace(' ', '\n') for t in target_cols]
cv_r2_values = [cv_results[t]['cv_r2'] for t in target_cols]
colors = ['green' if r2 > 0.3 else 'orange' if r2 > 0 else 'red' for r2 in cv_r2_values]

bars = ax.bar(targets_plot, cv_r2_values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Weak (R²=0.3)')
ax.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (R²=0.6)')

ax.set_ylabel('Cross-Validation R²', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Summary (63 samples, 6 features per target)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, cv_r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05 if height > 0 else height - 0.15,
            f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: model_performance_summary.png")

# ============================================================================
# 5. SAVE MODELS
# ============================================================================
with open('realistic_opv_models.pkl', 'wb') as f:
    pickle.dump({
        'models': final_models,
        'scalers': final_scalers,
        'selected_features': selected_features,
        'cv_results': {k: {kk: vv for kk, vv in v.items() if kk != 'y_pred_cv'}
                       for k, v in cv_results.items()}
    }, f)
print("\n💾 Models saved to: realistic_opv_models.pkl")

# ============================================================================
# 6. FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

print("\n📊 MODEL PERFORMANCE:")
for target in target_cols:
    cv_r2 = cv_results[target]['cv_r2']
    model_name = cv_results[target]['model_name']

    status = "❌ UNRELIABLE" if cv_r2 < 0 else \
        "⚠️  WEAK" if cv_r2 < 0.3 else \
            "⚙️  MODERATE" if cv_r2 < 0.6 else \
                "✅ GOOD"

    print(f"\n{target:20s}: {status}")
    print(f"   Model: {model_name}")
    print(f"   CV R²: {cv_r2:6.3f}")
    print(f"   Features: {', '.join(selected_features[target][:3])}...")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)

print("\n1. MODEL RELIABILITY:")
print("   ❌ R² < 0.0  : Worse than mean - DO NOT USE")
print("   ⚠️  R² 0.0-0.3: Weak - use only for rough screening")
print("   ⚙️  R² 0.3-0.6: Moderate - reasonable for ranking compounds")
print("   ✅ R² > 0.6  : Good - suitable for predictions (rare with 63 samples)")

print("\n2. PREDICTIONS UNCERTAINTY:")
for target in target_cols:
    cv_rmse = cv_results[target]['cv_rmse']
    target_std = y[target].std()
    uncertainty_pct = (cv_rmse / target_std) * 100
    print(f"   {target:20s}: ±{cv_rmse:.2f} ({uncertainty_pct:.0f}% of std dev)")

print("\n3. REALISTIC EXPECTATIONS:")
print("   • These models are at the LIMIT of what's possible with 63 samples")
print("   • Predictions have HIGH UNCERTAINTY - use as guides, not gospel")
print("   • Models work better for RANKING than absolute prediction")
print("   • Collect more data (150+ samples) for production-quality models")

print("\n4. RECOMMENDED USAGE:")
print("   ✅ Screening compound libraries (relative comparisons)")
print("   ✅ Identifying promising candidates for synthesis")
print("   ✅ Understanding feature importance trends")
print("   ❌ Precise quantitative predictions")
print("   ❌ Regulatory submissions or publications")
print("   ❌ Optimization without experimental validation")

print("\n" + "=" * 80)
print("ACTION ITEMS")
print("=" * 80)

print("\n1. IMMEDIATE: Use these models for preliminary screening only")
print("2. SHORT-TERM: Validate top predictions experimentally")
print("3. LONG-TERM: Collect 150-200 samples for robust models")
print("4. ALTERNATIVE: Explore transfer learning from public OPV databases")
print("5. PUBLICATION: Report uncertainty ranges, not just R² values")

print("\n" + "=" * 80)
print("✅ Analysis complete - models ready with realistic expectations")
print("=" * 80)