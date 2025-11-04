"""
Check that the simple decision tree for nonic fields gives 100% accuracy
for all nonic Galois extensions in the LMFDB.
"""
import polars as pl


def nonic_tree(a1000, a343, a27):
    """
    If a_1000 <= 4.5 or a_343 <= 0.5 or a_27 <= 0.5 then return "9T1",
    else return "9T2".
    """
    if a1000 <= 4.5 or a343 <= 0.5 or a27 <= 0.5:
        return "9T1"
    else:
        return "9T2"


if __name__ == "__main__":
    df_nonic = pl.read_csv("data_nf/nf_9_galois.csv")
    mismatches = []
    for row in df_nonic.iter_rows(named=True):
        a1000 = row["a_01000"]
        a343 = row["a_00343"]
        a27 = row["a_00027"]
        galois_label = row["galois_label"]
        prediction = nonic_tree(a1000, a343, a27)
        if prediction != galois_label:
            print("Mismatch found!")
            label = row["label"]
            print(f"For Label: {label}, Predicted: {prediction}, Actual: {galois_label}")
            mismatches.append(label)
    print(f"Total mismatches: {len(mismatches)}")
