"""
Prediction Script: Predict OPV Properties for New Molecules
============================================================
This script reads a CSV file with new molecular designs and predicts
their OPV properties using previously trained models.

CSV Format Required:
-------------------
Column name    | Description
---------------|---------------------------------------------------
D_SMILES       | Donor SMILES string (required)
D_Name         | Donor name (optional, for identification)
D_HOMO         | Donor HOMO energy level (eV)
D_LUMO         | Donor LUMO energy level (eV)
D_lamda_max    | Donor max absorption wavelength (nm)
A_HOMO         | Acceptor HOMO energy level (eV)
A_LUMO         | Acceptor LUMO energy level (eV)
A_lamda_max    | Acceptor max absorption wavelength (nm)

The script will predict:
- PCE (%) - Power Conversion Efficiency
- FF (%) - Fill Factor
- JSC (mA/cm²) - Short Circuit Current Density
- VOC (V) - Open Circuit Voltage
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

print("=" * 70)
print("OPV PROPERTY PREDICTION FOR NEW MOLECULES")
print("=" * 70)

# ============================================================================
# 1. LOAD TRAINED MODELS
# ============================================================================
print("\n📂 Loading trained models...")

model_files = ['opv_models_comparison.pkl', 'opv_models.pkl']
model_file = None

for fname in model_files:
    if os.path.exists(fname):
        model_file = fname
        break

if model_file is None:
    print("❌ ERROR: No trained model file found!")
    print("   Please run 'model_comparison.py' or 'heat_map_train_test_COMPLETE_FIX.py' first.")
    sys.exit(1)

with open(model_file, 'rb') as f:
    saved_data = pickle.load(f)

if 'best_models' in saved_data:
    # From comparison script
    models_dict = saved_data['best_models']
    feature_cols = saved_data['feature_cols']
    print(f"✅ Loaded best models from comparison: {model_file}")

    # Extract models and their names
    models = {}
    model_names = {}
    scalers = {}
    for target, data in models_dict.items():
        models[target] = data['model']
        model_names[target] = data['model_name']
        scalers[target] = data.get('scaler', None)
else:
    # From single model training
    models = saved_data['models']
    feature_cols = saved_data['feature_cols']
    model_names = {target: 'XGBoost' for target in models.keys()}
    scalers = {target: None for target in models.keys()}
    print(f"✅ Loaded XGBoost models: {model_file}")

target_cols = list(models.keys())
print(f"✅ Loaded {len(models)} models for: {', '.join(target_cols)}")

# ============================================================================
# 2. LOAD NEW MOLECULES FROM CSV
# ============================================================================
print("\n📂 Loading new molecules...")

# Get filename from command line or use default
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    # Default filename
    input_file = 'new_molecules.csv'
    print(f"   Using default file: {input_file}")
    print(f"   (Run with: python predict_properties.py your_file.csv)")

if not os.path.exists(input_file):
    print(f"\n❌ ERROR: File '{input_file}' not found!")
    print("\n📝 Please create a CSV file with the following columns:")
    print("   - D_SMILES (required)")
    print("   - D_Name (optional)")
    print("   - D_HOMO, D_LUMO, D_lamda_max")
    print("   - A_HOMO, A_LUMO, A_lamda_max")
    print("\n   Example:")
    print("   D_Name,D_SMILES,D_HOMO,D_LUMO,D_lamda_max,A_HOMO,A_LUMO,A_lamda_max")
    print("   NewMol1,CCCc1ccc(...),-5.2,-3.4,550,-5.65,-4.1,830")
    sys.exit(1)

# Try to read with different encodings and separators
try:
    new_data = pd.read_csv(input_file, encoding='latin1', sep=';')
    print(f"   Read with: encoding='latin1', sep=';'")
except:
    try:
        new_data = pd.read_csv(input_file, encoding='utf-8', sep=',')
        print(f"   Read with: encoding='utf-8', sep=','")
    except Exception as e:
        print(f"❌ ERROR reading file: {e}")
        sys.exit(1)

# Fix BOM if present
new_data.columns = [col.replace('ï»¿', '') for col in new_data.columns]

# Convert European number format if needed
numeric_cols = ['D_HOMO', 'D_LUMO', 'D_lamda_max', 'A_HOMO', 'A_LUMO', 'A_lamda_max']
for col in numeric_cols:
    if col in new_data.columns:
        if new_data[col].dtype == 'object':
            new_data[col] = new_data[col].astype(str).str.replace(',', '.').astype(float)

print(f"✅ Loaded {len(new_data)} new molecules")

# ============================================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================================
print("\n🔍 Validating data...")

required_cols = ['D_SMILES', 'D_HOMO', 'D_LUMO', 'D_lamda_max',
                 'A_HOMO', 'A_LUMO', 'A_lamda_max']

missing_cols = [col for col in required_cols if col not in new_data.columns]

if missing_cols:
    print(f"❌ ERROR: Missing required columns: {missing_cols}")
    print(f"   Available columns: {new_data.columns.tolist()}")
    sys.exit(1)

print("✅ All required columns present")

# Check for molecule names
if 'D_Name' not in new_data.columns:
    new_data['D_Name'] = [f"Molecule_{i + 1}" for i in range(len(new_data))]
    print("⚠️  No 'D_Name' column found, using default names")

# ============================================================================
# 4. CALCULATE MOLECULAR DESCRIPTORS
# ============================================================================
print("\n🧬 Calculating molecular descriptors...")


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
desc_df = new_data['D_SMILES'].apply(mol_descriptors).apply(pd.Series)
desc_df.columns = desc_names

# Check for invalid SMILES
invalid_smiles = desc_df.isnull().any(axis=1)
if invalid_smiles.any():
    invalid_names = new_data.loc[invalid_smiles, 'D_Name'].tolist()
    print(f"⚠️  Warning: Invalid SMILES found for: {invalid_names}")
    print("   These molecules will be skipped in predictions")

# Combine with original data
new_data_with_desc = pd.concat([new_data, desc_df], axis=1)

# Feature engineering
new_data_with_desc['D_Bandgap'] = new_data_with_desc['D_LUMO'] - new_data_with_desc['D_HOMO']
new_data_with_desc['A_Bandgap'] = new_data_with_desc['A_LUMO'] - new_data_with_desc['A_HOMO']
new_data_with_desc['HOMO_offset'] = new_data_with_desc['D_HOMO'] - new_data_with_desc['A_HOMO']
new_data_with_desc['LUMO_offset'] = new_data_with_desc['D_LUMO'] - new_data_with_desc['A_LUMO']
new_data_with_desc['Lambda_gap'] = new_data_with_desc['A_lamda_max'] - new_data_with_desc['D_lamda_max']

# Remove rows with invalid SMILES
new_data_clean = new_data_with_desc.dropna()
n_removed = len(new_data_with_desc) - len(new_data_clean)

if n_removed > 0:
    print(f"⚠️  Removed {n_removed} molecules with invalid SMILES")

print(f"✅ {len(new_data_clean)} molecules ready for prediction")

# ============================================================================
# 5. PREPARE FEATURES
# ============================================================================
print("\n🔧 Preparing features...")

# Extract feature columns (must match training data)
try:
    X_new = new_data_clean[feature_cols].copy()
    print(f"✅ Extracted {len(feature_cols)} features")
except KeyError as e:
    print(f"❌ ERROR: Missing feature column: {e}")
    print(f"   Required features: {feature_cols}")
    print(f"   Available columns: {new_data_clean.columns.tolist()}")
    sys.exit(1)

# ============================================================================
# 6. MAKE PREDICTIONS
# ============================================================================
print("\n🔮 Making predictions...")
print("=" * 70)

predictions = {}

for target in target_cols:
    model = models[target]
    model_name = model_names[target]
    scaler = scalers[target]

    # Scale features if needed (for KNN and Linear Regression)
    if scaler is not None:
        X_pred = scaler.transform(X_new)
    else:
        X_pred = X_new

    # Make predictions
    pred = model.predict(X_pred)
    predictions[target] = pred

    print(f"✅ {target} predicted using {model_name}")

# ============================================================================
# 7. CREATE RESULTS DATAFRAME
# ============================================================================
print("\n📊 Generating results...")

results_df = new_data_clean[['D_Name', 'D_SMILES']].copy()

for target in target_cols:
    results_df[f'Predicted_{target}'] = predictions[target]

# Add input features for reference
for col in ['D_HOMO', 'D_LUMO', 'A_HOMO', 'A_LUMO', 'D_lamda_max', 'A_lamda_max']:
    if col in new_data_clean.columns:
        results_df[col] = new_data_clean[col].values

# ============================================================================
# 8. DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)

for idx, row in results_df.iterrows():
    print(f"\n🔬 {row['D_Name']}:")
    print(f"   SMILES: {row['D_SMILES'][:60]}...")
    print(f"   Predicted Properties:")
    print(f"      PCE:  {row['Predicted_PCE (%)']:.2f} %")
    print(f"      VOC:  {row['Predicted_VOC (V)']:.3f} V")
    print(f"      JSC:  {row['Predicted_JSC (mA/cmÂ²)']:.2f} mA/cm²")
    print(f"      FF:   {row['Predicted_FF (%)']:.2f} %")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
output_file = input_file.replace('.csv', '_predictions.csv')

# Save with standard formatting
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print(f"✅ Results saved to: {output_file}")
print("=" * 70)

# ============================================================================
# 10. SUMMARY STATISTICS
# ============================================================================
print("\n📊 Prediction Summary Statistics:")
print("=" * 70)

summary_stats = []
for target in target_cols:
    pred_col = f'Predicted_{target}'
    summary_stats.append({
        'Property': target,
        'Mean': results_df[pred_col].mean(),
        'Std': results_df[pred_col].std(),
        'Min': results_df[pred_col].min(),
        'Max': results_df[pred_col].max()
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

# Find best candidate
best_pce_idx = results_df['Predicted_PCE (%)'].idxmax()
best_molecule = results_df.loc[best_pce_idx]

print("\n🏆 Best Predicted Candidate (Highest PCE):")
print(f"   Name: {best_molecule['D_Name']}")
print(f"   Predicted PCE: {best_molecule['Predicted_PCE (%)']:.2f}%")
print(f"   Predicted VOC: {best_molecule['Predicted_VOC (V)']:.3f} V")
print(f"   Predicted JSC: {best_molecule['Predicted_JSC (mA/cmÂ²)']:.2f} mA/cm²")
print(f"   Predicted FF:  {best_molecule['Predicted_FF (%)']:.2f}%")

print("\n" + "=" * 70)
print("✅ PREDICTION COMPLETE!")
print("=" * 70)