"""
Standalone Script: Save SMILES Structures as PNG Images
========================================================
This script reads the CSV file and saves each molecular structure
from the D_SMILES column as a separate PNG image.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("📂 Loading data...")
data = pd.read_csv('data_aromatic_converted.csv', encoding='latin1', sep=';')

# Fix BOM (Byte Order Mark) in column names
# When reading UTF-8 BOM with latin1 encoding, it appears as 'ï»¿'
data.columns = [col.replace('ï»¿', '') for col in data.columns]

print(f"✅ Loaded {len(data)} molecules")
print(f"\n📋 Available columns:")
print(data.columns.tolist())

# Verify D_Name exists
if 'D_Name' not in data.columns:
    print("\n❌ ERROR: 'D_Name' column not found!")
    print(f"Available columns: {data.columns.tolist()}")
    exit(1)

# ============================================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================================
output_dir = 'claude_input_struct'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✅ Created directory: {output_dir}")
else:
    print(f"✅ Using existing directory: {output_dir}")

# ============================================================================
# 3. SAVE EACH SMILES AS PNG
# ============================================================================
print("\n🎨 Generating molecular structure images...")
print("=" * 60)

saved_count = 0
failed_count = 0
failed_molecules = []

for idx, row in data.iterrows():
    smiles = row['D_SMILES']
    donor_name = row['D_Name']

    # Clean the name to make it filename-safe
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(donor_name))

    try:
        # Generate molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Generate image with good quality
            # size=(800, 600) creates a nice large image
            img = Draw.MolToImage(mol, size=(800, 600))

            # Save with zero-padded index and donor name
            filename = f"{output_dir}/{idx:03d}_{safe_name}.png"
            img.save(filename)

            saved_count += 1
            print(f"  ✅ [{idx+1:2d}/{len(data)}] Saved: {safe_name}")
        else:
            failed_count += 1
            failed_molecules.append((idx, donor_name, "Invalid SMILES"))
            print(f"  ⚠️  [{idx+1:2d}/{len(data)}] Failed: {donor_name} - Could not parse SMILES")

    except Exception as e:
        failed_count += 1
        failed_molecules.append((idx, donor_name, str(e)))
        print(f"  ❌ [{idx+1:2d}/{len(data)}] Error: {donor_name} - {str(e)}")

# ============================================================================
# 4. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✅ Successfully saved: {saved_count} structures")
print(f"❌ Failed to save: {failed_count} structures")
print(f"📁 Output directory: {output_dir}/")
print("=" * 60)

if failed_molecules:
    print("\n⚠️  Failed molecules:")
    for idx, name, reason in failed_molecules:
        print(f"  - [{idx}] {name}: {reason}")

# ============================================================================
# 5. EXAMPLE FILENAMES
# ============================================================================
if saved_count > 0:
    print("\n📋 Example filenames generated:")
    image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])[:5]
    for filename in image_files:
        print(f"  - {filename}")
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more")

print("\n✅ Complete! All molecular structures saved.")