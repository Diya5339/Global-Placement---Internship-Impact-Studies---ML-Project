"""Quick test to verify the regression mask fix"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a minimal test dataframe with placement data
test_data = pd.DataFrame({
    'placement_status': ['Placed', 'Not Placed', 'Placed', 'Not Placed', 'Placed'],
    'salary': [80000, 0, 90000, 0, 100000]
})

print("Test DataFrame:")
print(test_data)
print("\nUnique values:", test_data['placement_status'].unique())

# Simulate the label encoder (what happens in the real script)
label_encoder = LabelEncoder()
encoded = label_encoder.fit_transform(test_data['placement_status'])
print("\nLabelEncoder classes:", label_encoder.classes_)
print("Encoded values:", encoded)

# Test the new mask logic (what we fixed)
placed_class_name = None
for cls_name in label_encoder.classes_:
    if cls_name.lower() == "placed":
        placed_class_name = cls_name
        break

if placed_class_name is None:
    placed_class_name = "Placed"

print(f"\nPlaced class name found: '{placed_class_name}'")

# Create mask using the fixed logic
placed_mask = test_data['placement_status'] == placed_class_name
print(f"Mask: {placed_mask.values}")

# Filter
df_reg = test_data.loc[placed_mask].copy()
print(f"\nFiltered DataFrame (placed only):")
print(df_reg)
print(f"Shape: {df_reg.shape}")

if df_reg.shape[0] > 0:
    print("\n✓ SUCCESS: Mask correctly identified placed students!")
else:
    print("\n✗ FAILED: Mask returned 0 rows!")
