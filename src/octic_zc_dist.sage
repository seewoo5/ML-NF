"""
Compute distribution of square-indexed zeta coefficients for octic number fields.
"""
import pathlib

from utils import *
from tqdm import tqdm


current_dir = pathlib.Path(__file__).resolve().parent


if __name__ == "__main__":
    df = pl.read_csv(current_dir / "data_nf/nf_8_galois.csv", schema_overrides={f"c_{i}": pl.Int128 for i in range(8)})
    df_8T1 = df.filter(pl.col("galois_label") == "8T1")
    df_8T2 = df.filter(pl.col("galois_label") == "8T2")
    df_8T3 = df.filter(pl.col("galois_label") == "8T3")
    df_8T4 = df.filter(pl.col("galois_label") == "8T4")
    df_8T5 = df.filter(pl.col("galois_label") == "8T5")

    # Nonabelian octic fields, 8T4 and 8T5
    print("==== Nonabelian octic fields ====")
    indices = [2 ** 2, 3 ** 2, 5 ** 2, 7 ** 2, 11 ** 2, 13 ** 2, 23 ** 2]
    for n in indices:
        print(f"Zeta coefficient n = {n}")
        print("==== 8T4 ====")
        zeta_count(df_8T4, n)
        print("==== 8T5 ====")
        zeta_count(df_8T5, n)

    print("==== p^4-th indices, all octic fields ====")
    indices = [2 ** 4, 3 ** 4, 5 ** 4, 7 ** 4]
    for n in indices:
        print(f"Zeta coefficient n = {n}")
        print("==== 8T1 ====")
        zeta_count(df_8T1, n)
        print("==== 8T2 ====")
        zeta_count(df_8T2, n)
        print("==== 8T3 ====")
        zeta_count(df_8T3, n)
        print("==== 8T4 ====")
        zeta_count(df_8T4, n)
        print("==== 8T5 ====")
        zeta_count(df_8T5, n)
