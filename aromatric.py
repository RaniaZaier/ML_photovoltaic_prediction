import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os


def read_csv_auto_encoding(filepath):
    """Read CSV with automatic encoding detection"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
    separators = [';', ',', '\t']

    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                # Check if we got multiple columns
                if len(df.columns) > 1:
                    print(f"✅ CSV loaded successfully")
                    print(f"   Encoding: {encoding}")
                    print(f"   Separator: '{sep}'")
                    return df, sep
            except:
                continue

    raise ValueError(f"Could not read {filepath}")


def convert_to_aromatic(smi):
    """Convert SMILES to aromatic notation"""
    if pd.isna(smi) or not str(smi).strip():
        return smi, "⚠️ Empty"

    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            # Try without sanitization
            mol = Chem.MolFromSmiles(str(smi), sanitize=False)
            if mol is None:
                return smi, "❌ Parse failed"
            Chem.SanitizeMol(mol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)

        # Generate canonical aromatic SMILES
        aromatic_smi = Chem.MolToSmiles(
            mol,
            kekuleSmiles=False,
            canonical=True,
            isomericSmiles=True
        )
        return aromatic_smi, "✅ OK"

    except Exception as e:
        return smi, f"❌ Error"


def save_molecule_as_png(smiles, filename, output_dir, img_size=(300, 300)):
    """Save a molecule structure as PNG image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Generate 2D coordinates
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)

        # Draw molecule
        img = Draw.MolToImage(mol, size=img_size)

        # Save image
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        return True
    except Exception as e:
        print(f"   ⚠️ Failed to save image for {filename}: {str(e)}")
        return False


def main():
    # Configuration
    input_csv = "data_cleaned.csv"
    output_csv = "data_cleande_aromatic.csv"
    smiles_column = "D_SMILES"
    output_images_dir = "molecule_images"  # Folder for PNG images
    img_size = (400, 400)  # Image size (width, height) in pixels

    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} not found in current directory!")
        print(f"Current directory: {os.getcwd()}")
        return

    # Create output directory for images
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
        print(f"📁 Created directory: {output_images_dir}/")
    else:
        print(f"📁 Using existing directory: {output_images_dir}/")

    # Read CSV
    print(f"\n📂 Reading {input_csv}...")
    df, original_sep = read_csv_auto_encoding(input_csv)
    print(f"✅ Loaded {len(df)} molecules")
    print(f"📋 Columns ({len(df.columns)}): {', '.join(df.columns)}")

    # Check if SMILES column exists
    if smiles_column not in df.columns:
        print(f"\n❌ Column '{smiles_column}' not found!")
        print(f"Available columns: {list(df.columns)}")
        return

    # Convert SMILES
    print(f"\n🔄 Converting SMILES to canonical aromatic notation...")
    print(f"{'=' * 70}")

    results = df[smiles_column].apply(convert_to_aromatic)

    # Replace the SMILES column with converted SMILES
    df[smiles_column] = results.apply(lambda x: x[0])

    # Add a status column
    df["Conversion_Status"] = results.apply(lambda x: x[1])

    # Statistics
    statuses = results.apply(lambda x: x[1])
    print(f"\n📊 Conversion Statistics:")
    status_counts = statuses.value_counts()
    for status, count in status_counts.items():
        print(f"   {status}: {count}")

    # Save molecular structures as PNG images
    print(f"\n🖼️  Generating PNG images for all molecules...")
    print(f"{'=' * 70}")

    successful_images = 0
    failed_images = 0

    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        status = row["Conversion_Status"]

        # Only save images for successfully converted molecules
        if status == "✅ OK" and pd.notna(smiles):
            # Create filename: molecule_001.png, molecule_002.png, etc.
            filename = f"molecule_{idx + 1:04d}.png"

            if save_molecule_as_png(smiles, filename, output_images_dir, img_size):
                successful_images += 1
                if (idx + 1) % 100 == 0:  # Progress update every 100 molecules
                    print(f"   Processed {idx + 1}/{len(df)} molecules...")
            else:
                failed_images += 1
        else:
            failed_images += 1

    print(f"\n📊 Image Generation Statistics:")
    print(f"   ✅ Successfully saved: {successful_images} images")
    print(f"   ❌ Failed/Skipped: {failed_images} molecules")
    print(f"   📁 Images saved to: {output_images_dir}/")

    # Save CSV result with same separator as input
    df.to_csv(output_csv, index=False, sep=original_sep, encoding='utf-8-sig')
    print(f"\n💾 CSV results saved to: {output_csv}")
    print(f"📋 Output columns ({len(df.columns)}): {', '.join(df.columns)}")
    print(f"📋 Separator used: '{original_sep}'")

    # Show example
    print(f"\n📝 First molecule (showing all columns):")
    print(f"{'-' * 70}")
    first_row = df.iloc[0]
    for col in df.columns:
        value = str(first_row[col])
        display_value = value[:60] + "..." if len(value) > 60 else value
        print(f"   {col:20s}: {display_value}")

    print(f"\n{'=' * 70}")
    print(f"✅ Conversion complete!")
    print(f"📄 Input:  {len(df)} molecules with {len(df.columns) - 1} columns")
    print(f"📄 Output: {len(df)} molecules with {len(df.columns)} columns")
    print(f"🖼️  Images: {successful_images} PNG files in '{output_images_dir}/'")
    print(f"💡 SMILES column has been replaced with canonical aromatic SMILES")
    print(f"💡 All other columns have been preserved")


if __name__ == "__main__":
    main()