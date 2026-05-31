import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings

warnings.filterwarnings("ignore")

print("="*80)
print("DIAGNOSTIC: Check Data and Features")
print("="*80)

# Load data
data = pd.read_csv('data_cleande_aromatic.csv', encoding='latin1', sep=';')
data.columns = [col.replace('ï»¿', '') for col in data.columns]

print(f"\n1️⃣ INITIAL DATA CHECK")
print(f"   Rows: {len(data)}")
print(f"   Columns: {len(data.columns)}")
print(f"\n   Column names:")
for col in data.columns:
    print(f"      - {col}")

# Convert numeric columns
numeric_columns = ['D_HOMO', 'D_LUMO', 'D_lamda_max', 'A_HOMO', 'A_LUMO',
                   'A_lamda_max', 'VOC (V)', 'JSC (mA/cmÂ²)', 'FF (%)', 'PCE (%)']

for col in numeric_columns:
    if col in data.columns:
        data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Fix FF
if (data['FF (%)'] < 10).any():
    data.loc[data['FF (%)'] < 1, 'FF (%)'] *= 100
    data.loc[(data['FF (%)'] >= 1) & (data['FF (%)'] < 10), 'FF (%)'] *= 10

print(f"\n2️⃣ AFTER NUMBER CONVERSION")
print(f"   Rows: {len(data)}")

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

print(f"\n3️⃣ AFTER DESCRIPTOR CALCULATION")
print(f"   Descriptor columns: {desc_names}")
print(f"   Descriptor rows: {len(desc_df)}")
print(f"   NaN count per column:")
for col in desc_names:
    nan_count = desc_df[col].isna().sum()
    if nan_count > 0:
        print(f"      {col}: {nan_count} NaN values")

# Combine
data_full = pd.concat([data, desc_df], axis=1)
print(f"\n4️⃣ AFTER COMBINING")
print(f"   Total columns: {len(data_full.columns)}")
print(f"   Total rows: {len(data_full)}")

# Check for NaN
print(f"\n   Columns with NaN:")
nan_cols = data_full.columns[data_full.isna().any()].tolist()
for col in nan_cols:
    nan_count = data_full[col].isna().sum()
    print(f"      {col}: {nan_count} NaN values")

# Drop NaN
data_clean = data_full.dropna()
print(f"\n5️⃣ AFTER DROPPING NaN")
print(f"   Rows remaining: {len(data_clean)}")

if len(data_clean) == 0:
    print("\n   ❌ ERROR: All rows were dropped!")
    print("   This means every row has at least one NaN value")
    print("\n   Checking which columns caused the issue:")
    for col in data_full.columns:
        non_nan = data_full[col].notna().sum()
        if non_nan < len(data_full):
            print(f"      {col}: {len(data_full) - non_nan} rows with NaN")
else:
    # Feature engineering
    data_clean['D_Bandgap'] = data_clean['D_LUMO'] - data_clean['D_HOMO']
    data_clean['A_Bandgap'] = data_clean['A_LUMO'] - data_clean['A_HOMO']
    data_clean['HOMO_offset'] = data_clean['D_HOMO'] - data_clean['A_HOMO']
    data_clean['LUMO_offset'] = data_clean['D_LUMO'] - data_clean['A_LUMO']
    
    print(f"\n6️⃣ AFTER FEATURE ENGINEERING")
    print(f"   Total columns: {len(data_clean.columns)}")
    print(f"   Total rows: {len(data_clean)}")
    
    # Check if expected features exist
    expected_features = [
        'D_HOMO', 'D_LUMO', 'D_lamda_max',
        'A_HOMO', 'A_LUMO', 'A_lamda_max',
        'MolWt', 'TPSA', 'NumHAcceptors',
        'NumRotatableBonds', 'MolLogP', 'RingCount',
        'D_Bandgap', 'A_Bandgap',
        'HOMO_offset', 'LUMO_offset'
    ]
    
    print(f"\n7️⃣ CHECKING EXPECTED FEATURES")
    missing = []
    present = []
    for feat in expected_features:
        if feat in data_clean.columns:
            present.append(feat)
        else:
            missing.append(feat)
    
    print(f"   ✅ Present ({len(present)}): {', '.join(present[:5])}...")
    if missing:
        print(f"   ❌ Missing ({len(missing)}): {', '.join(missing)}")
    
    # Test feature selection
    target_cols = ['PCE (%)', 'FF (%)', 'JSC (mA/cmÂ²)', 'VOC (V)']
    
    print(f"\n8️⃣ TESTING FEATURE SELECTION")
    X_test = data_clean[present]  # Use only present features
    y_test = data_clean[target_cols]
    
    print(f"   X shape: {X_test.shape}")
    print(f"   y shape: {y_test.shape}")
    
    # Test correlation
    for target in target_cols:
        corr = X_test.corrwith(y_test[target]).abs().sort_values(ascending=False)
        top_3 = corr.head(3).index.tolist()
        print(f"\n   {target} - Top 3 correlations:")
        for feat in top_3:
            print(f"      {feat}: {corr[feat]:.3f}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

if len(data_clean) == 0:
    print("\n❌ PROBLEM: No data remaining after cleaning")
    print("   → Check for NaN values in your CSV file")
    print("   → Some SMILES might be invalid")
    print("   → Some columns might have formatting issues")
else:
    print(f"\n✅ Data looks good: {len(data_clean)} samples ready")
    print("   → You can proceed with model training")
