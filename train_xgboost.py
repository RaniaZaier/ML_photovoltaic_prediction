import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv('data-aromatic-param-all.csv', encoding='latin1', sep=';')

# Function to calculate molecular descriptors from SMILES
# Function to calculate molecular descriptors from SMILES
def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None]*7
    return [
        Descriptors.MolWt(mol),               # Molecular Weight
        Descriptors.TPSA(mol),                # Topological Polar Surface Area
        Descriptors.MolMR(mol),               # Molar Refractivity (replaces NumHDonors)
        Descriptors.NumHAcceptors(mol),      # Number of H-bond acceptors
        Descriptors.NumRotatableBonds(mol),  # Number of rotatable bonds
        Descriptors.MolLogP(mol),             # Octanol-water partition coefficient
        Descriptors.RingCount(mol)            # Number of rings
    ]

# Update descriptor names accordingly
desc_names = ['MolWt', 'TPSA', 'MolMR', 'NumHAcceptors', 'NumRotatableBonds', 'MolLogP', 'RingCount']
desc_df = data['SMILES'].apply(mol_descriptors).apply(pd.Series)
desc_df.columns = desc_names

# Combine descriptors with original numeric data (dropping SMILES)
data_numeric = pd.concat([data.drop(columns=['SMILES']), desc_df], axis=1)

# Drop rows with missing values
data_numeric = data_numeric.dropna()

# Features (all except targets)
feature_cols = [col for col in data_numeric.columns if col not in ['PCE (%)', 'FF (%)', 'JSC (mA/cm²)', 'VOC (V)']]
X = data_numeric[feature_cols]

# Targets: photovoltaic parameters
y = data_numeric[['PCE (%)', 'FF (%)', 'JSC (mA/cm²)', 'VOC (V)']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- XGBoost training (active) ---
models = {}
y_pred = np.zeros(y_test.shape)

for i, target in enumerate(y.columns):
    model = XGBRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train[target])
    y_pred[:, i] = model.predict(X_test)
    models[target] = model

"""
# --- Random Forest training (commented) ---
models_rf = {}
y_pred_rf = np.zeros(y_test.shape)

for i, target in enumerate(y.columns):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train[target])
    y_pred_rf[:, i] = rf.predict(X_test)
    models_rf[target] = rf
"""

# Evaluate each target for XGBoost
for i, target_name in enumerate(y.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{target_name} - Test MSE: {mse:.4f}, R^2: {r2:.4f}")

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = data_numeric.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap of Features, Molecular Descriptors, and Targets')
plt.savefig('heat_map.png', dpi=300)
plt.show()


# Plot true vs predicted in 2x2 grid with smaller figure and labels
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, target_name in enumerate(y.columns):
    ax = axes[i]
    ax.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.7, s=20)
    ax.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
            'r--', lw=1.5)
    ax.set_xlabel(f'True {target_name}', fontsize=9)
    ax.set_ylabel(f'Predicted {target_name}', fontsize=9)
    ax.set_title(f'True vs Predicted: {target_name}', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True)

plt.tight_layout()
plt.savefig('train_test.png', dpi=300)
plt.show()
  for this code give a name of repository in github and a quick organized readme, note that i am spliting the  work in my github, reposetory for cleaning data , one for training ML etc etc, to make it clear
