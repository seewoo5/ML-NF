"""
Verifies the theorems on decomposition type and the possible values of zeta coefficients in the paper for the number fields in LMFDB.
"""
import pathlib

from utils import *
from tqdm import tqdm


current_dir = pathlib.Path(__file__).resolve().parent


def check_value_in(df: pl.DataFrame, index: int, values: List):
    col_values = df[f"a_{index:05d}"].unique().to_list()
    assert set(col_values) <= set(values), f"Invalid values in column a_{index:05d}: {col_values} not in {values}"


def count_and_check_decomposition_types(
    df: pl.DataFrame,
    degree: int,
    p: int,
    decomp_types: List,
    random_sample: Optional[int] = None,
):
    cnt = {t: 0 for t in decomp_types}
    R.<x> = PolynomialRing(QQ)
    if random_sample is not None:
        df = df.sample(n=random_sample)
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        # Construct number field from defining polynomial
        poly = x ** degree
        for i in range(degree):
            poly += Rational(row[f"c_{i}"]) * x ** i
        K.<a> = NumberField(poly)

        # Check decomposition of p in K has the expected type
        dec_types = K.decomposition_type(p)
        assert len(dec_types) == 1, f"{row['label']} is not Galois over Q, {dec_types}"
        e, f, g = dec_types[0]
        assert (e, f, g) in decomp_types, f"Decomposition type {(e, f, g)} not in {decomp_types}"
        cnt[(e, f, g)] += 1
    print(f"Counts: {cnt}")


def generate_primes_mod(a, b, cnt = 3):
    """
    Generate a list of primes p such that p = a mod b.
    If b == 0, generate all primes p = a.
    """
    if b == 0:
        return [a]
    primes = []
    for p in range(a, 10000, b):
        if is_prime(p):
            primes.append(p)
        if len(primes) >= cnt:
            break
    return primes


def verify_galois(
    degree: int,
    dec_types: dict[tuple[str, int, int], List[tuple[int, int, int]]],
    zc: dict[tuple[str, int], List[int]],
    random_sample: Optional[int] = None,
):
    # Load the data
    schema = {f"c_{i}": pl.Int128 for i in range(degree)}
    if degree == 10:
        df = pl.read_csv(current_dir / f"data_nf/nf_10_galois_sq_cb_qu_100000.csv", schema_overrides=schema)
    else:
        df = pl.read_csv(current_dir / f"data_nf/nf_{degree}_galois.csv", schema_overrides=schema)
    df_by_label = {}
    for galois_label, _, _ in dec_types.keys():
        if galois_label not in df_by_label:
            df_by_label[galois_label] = df.filter(pl.col("galois_label") == galois_label)

    # Check decomposition types
    for (galois_label, a, b), decomp_types in dec_types.items():
        primes = generate_primes_mod(a, b)
        for p in primes:
            if b == 0:
                print(f"{galois_label}, p = {p}")
            else:
                print(f"{galois_label}, p = {p} ({a} mod {b})")
            count_and_check_decomposition_types(
                df_by_label[galois_label],
                degree=degree,
                p=p,
                decomp_types=decomp_types,
                random_sample=random_sample,
            )

    # Check possible zeta coefficient values
    for (galois_label, n), zeta_coeffs in zc.items():
        print(f"{galois_label}, n = {n}")
        check_value_in(df_by_label[galois_label], n, zeta_coeffs)


# Decomposition types to be checked
# Key is (galois_label, a, b), where p = a if b == 0 else p == a mod b
# Values are lists of tuples (e, f, g) representing possible decomposition types
dec_types_quartic = {
    ("4T1", 2, 0): [(1, 1, 4), (1, 2, 2), (1, 4, 1), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    ("4T1", 1, 4): [(1, 1, 4), (1, 2, 2), (1, 4, 1), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    ("4T1", 3, 4): [(1, 1, 4), (1, 2, 2), (1, 4, 1), (2, 1, 2), (2, 2, 1)],
    ("4T2", 2, 0): [(1, 1, 4), (1, 2, 2), (2, 1, 2), (2, 2, 1), (4, 1, 1)],
    ("4T2", 1, 2): [(1, 1, 4), (1, 2, 2), (2, 1, 2), (2, 2, 1)],
}

def_types_nonic = {
    ("9T1", 3, 0): [(1, 1, 9), (1, 3, 3), (1, 9, 1), (3, 1, 3), (3, 3, 1), (9, 1, 1)],
    ("9T1", 1, 9): [(1, 1, 9), (1, 3, 3), (1, 9, 1), (3, 1, 3), (3, 3, 1), (9, 1, 1)],
    ("9T1", 4, 9): [(1, 1, 9), (1, 3, 3), (1, 9, 1), (3, 1, 3), (3, 3, 1)],
    ("9T1", 7, 9): [(1, 1, 9), (1, 3, 3), (1, 9, 1), (3, 1, 3), (3, 3, 1)],
    ("9T1", 2, 3): [(1, 1, 9), (1, 3, 3), (1, 9, 1)],
    ("9T2", 3, 0): [(1, 1, 9), (1, 3, 3), (3, 1, 3), (3, 3, 1)],
    ("9T2", 1, 3): [(1, 1, 9), (1, 3, 3), (3, 1, 3), (3, 3, 1)],
    ("9T2", 2, 3): [(1, 1, 9), (1, 3, 3)],
}

dec_types_sextic = {
    ("6T1", 2, 0): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (1, 6, 1), (2, 1, 3), (2, 3, 1)],
    ("6T1", 3, 0): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (1, 6, 1), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1), (6, 1, 1)],
    ("6T1", 1, 6): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (1, 6, 1), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1), (6, 1, 1)],
    ("6T1", 5, 6): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (1, 6, 1), (2, 1, 3), (2, 3, 1)],
    ("6T2", 2, 0): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)],
    ("6T2", 3, 0): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 1, 2), (3, 2, 1), (6, 1, 1)],
    ("6T2", 1, 6): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 1, 2)],
    ("6T2", 5, 6): [(1, 1, 6), (1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 2, 1)],
}

dec_types_octic = {
    ("8T1", 2, 0): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 1, 2), (8, 1, 1)],
    ("8T1", 1, 8): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (1, 8, 1), (2, 1, 4), (2, 2, 2), (2, 4, 1), (4, 1, 2), (4, 2, 1), (8, 1, 1)],
    ("8T1", 5, 8): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (1, 8, 1), (2, 1, 4), (2, 2, 2), (2, 4, 1), (4, 1, 2), (4, 2, 1)],
    ("8T1", 3, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (1, 8, 1), (2, 1, 4), (2, 2, 2), (2, 4, 1)],
    ("8T2", 2, 0): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (2, 4, 1), (4, 1, 2), (4, 2, 1), (8, 1, 1)],
    ("8T2", 1, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (2, 4, 1), (4, 1, 2), (4, 2, 1)],
    ("8T2", 3, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (2, 4, 1)],
    ("8T3", 2, 0): [(1, 1, 8), (1, 2, 4), (2, 1, 4), (2, 2, 2), (4, 1, 2), (4, 2, 1)],
    ("8T3", 1, 4): [(1, 1, 8), (1, 2, 4), (2, 1, 4), (2, 2, 2)],
    ("8T3", 3, 4): [(1, 1, 8), (1, 2, 4), (2, 1, 4), (2, 2, 2)],
    ("8T4", 2, 0): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 1, 2), (4, 2, 1), (8, 1, 1)],
    ("8T4", 1, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 1, 2)],
    ("8T4", 3, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 2, 1)],
    ("8T5", 2, 0): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 1, 2), (4, 2, 1), (8, 1, 1)],
    ("8T5", 1, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 1, 2)],
    ("8T5", 3, 4): [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 2, 2), (4, 2, 1)],
}

dec_types_decic = {
    ("10T1", 2, 0): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (1, 10, 1), (2, 1, 5), (2, 5, 1)],
    ("10T1", 1, 10): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (1, 10, 1), (2, 1, 5), (2, 5, 1), (5, 1, 2), (5, 2, 1), (10, 1, 1)],
    ("10T1", 9, 10): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (1, 10, 1), (2, 1, 5), (2, 5, 1)],
    ("10T2", 2, 0): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (2, 1, 5)],
    ("10T2", 5, 0): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (2, 1, 5), (5, 1, 2), (5, 2, 1), (10, 1, 1)],
    ("10T2", 1, 10): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (2, 1, 5), (5, 1, 2)],
    ("10T2", 9, 10): [(1, 1, 10), (1, 2, 5), (1, 5, 2), (2, 1, 5), (5, 2, 1)],
}

# Zeta coefficients to be checked
# Key is (galois_label, n), where n is the index of the zeta coefficient
zc_quartic = {
    ("4T1", 2 ** 2): [0, 1, 2, 3, 10],
    ("4T2", 2 ** 2): [1, 2, 3, 10],
    ("4T1", 3 ** 2): [0, 1, 2, 3, 10],
    ("4T2", 3 ** 2): [1, 2, 3, 10],
    ("4T1", 5 ** 2): [0, 1, 2, 3, 10],
    ("4T2", 5 ** 2): [1, 2, 3, 10],
}

zc_nonic = {
    ("9T1", 2 ** 3): [0, 3, 165],
    ("9T2", 2 ** 3): [3, 165],
    ("9T1", 3 ** 3): [0, 1, 3, 10, 165],
    ("9T2", 3 ** 3): [1, 3, 10, 165],
    ("9T1", 5 ** 3): [0, 3, 165],
    ("9T2", 5 ** 3): [3, 165],
}

zc_sextic = {
    ("6T1", 2 ** 2): [0, 3, 6, 21],
    ("6T2", 2 ** 2): [0, 1, 3, 6, 21],
    ("6T1", 5 ** 2): [0, 3, 6, 21],
    ("6T2", 5 ** 2): [0, 1, 3, 6, 21],
    ("6T1", 7 ** 2): [0, 1, 3, 6, 21],
    ("6T2", 7 ** 2): [0, 3, 6, 21],
    ("6T1", 2 ** 3): [0, 1, 2, 10, 56],
    ("6T2", 2 ** 3): [0, 2, 10, 56],
    ("6T1", 5 ** 3): [0, 1, 2, 4, 10, 56],
    ("6T2", 5 ** 3): [0, 2, 4, 10, 56],
    ("6T1", 7 ** 3): [0, 1, 2, 4, 10, 56],
    ("6T2", 7 ** 3): [0, 2, 4, 10, 56],
}

zc_octic = {
    ("8T1", 2 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T2", 2 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T3", 2 ** 2): [1, 2, 3, 4, 10, 36],
    ("8T4", 2 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T5", 2 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T1", 3 ** 2): [0, 2, 4, 10, 36],
    ("8T2", 3 ** 2): [0, 2, 4, 10, 36],
    ("8T3", 3 ** 2): [2, 4, 10, 36],
    ("8T4", 3 ** 2): [0, 1, 2, 4, 10, 36],
    ("8T5", 3 ** 2): [0, 1, 2, 4, 10, 36],
    ("8T1", 5 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T2", 5 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T3", 5 ** 2): [2, 4, 10, 36],
    ("8T4", 5 ** 2): [0, 2, 3, 4, 10, 36],
    ("8T5", 5 ** 2): [0, 2, 3, 4, 10, 36],
    ("8T1", 7 ** 2): [0, 2, 4, 10, 36],
    ("8T2", 7 ** 2): [0, 2, 4, 10, 36],
    ("8T3", 7 ** 2): [2, 4, 10, 36],
    ("8T4", 7 ** 2): [0, 1, 2, 4, 10, 36],
    ("8T5", 7 ** 2): [0, 1, 2, 4, 10, 36],
    ("8T1", 13 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T2", 13 ** 2): [0, 1, 2, 3, 4, 10, 36],
    ("8T3", 13 ** 2): [2, 4, 10, 36],
    ("8T4", 13 ** 2): [0, 2, 3, 4, 10, 36],
    ("8T5", 13 ** 2): [0, 2, 3, 4, 10, 36],
}

zc_decic = {
    ("10T1", 2 ** 2): [0, 5, 15, 55],
    ("10T2", 2 ** 2): [0, 5, 15, 55],
    ("10T1", 5 ** 2): [0, 1, 3, 5, 15, 55],
    ("10T2", 5 ** 2): [0, 1, 3, 5, 15, 55],
    ("10T1", 11 ** 2): [0, 1, 3, 5, 15, 55],
    ("10T2", 11 ** 2): [0, 3, 5, 15, 55],
    ("10T1", 19 ** 2): [0, 5, 15, 55],
    ("10T2", 19 ** 2): [0, 1, 5, 15, 55],
    ("10T1", 2 ** 3): [0, 35, 220],
    ("10T2", 2 ** 3): [0, 35, 220],
    ("10T1", 5 ** 3): [0, 1, 4, 35, 220],
    ("10T2", 5 ** 3): [0, 1, 4, 35, 220],
    ("10T1", 11 ** 3): [0, 1, 4, 35, 220],
    ("10T2", 11 ** 3): [0, 4, 35, 220],
    ("10T1", 19 ** 3): [0, 35, 220],
    ("10T2", 19 ** 3): [0, 35, 220],
    ("10T1", 2 ** 5): [0, 1, 2, 126, 2002],
    ("10T2", 2 ** 5): [0, 2, 126, 2002],
    ("10T1", 3 ** 5): [0, 1, 2, 126, 2002],
    ("10T2", 3 ** 5): [0, 2, 126, 2002],
    ("10T1", 5 ** 5): [0, 1, 2, 6, 126, 2002],
    ("10T2", 5 ** 5): [0, 1, 2, 6, 126, 2002],
    ("10T1", 7 ** 5): [0, 1, 2, 126, 2002],
    ("10T2", 7 ** 5): [0, 2, 126, 2002],
}


if __name__ == "__main__":
    # Whole script may take few hours to run. You can decrease the number of random samples to speed it up.
    random_sample = 5000
    print("==== Quartic fields ====")
    verify_galois(
        degree=4,
        dec_types=dec_types_quartic,
        zc=zc_quartic,
        random_sample=random_sample,
    )

    print("==== Nonic fields ====")
    verify_galois(
        degree=9,
        dec_types=def_types_nonic,
        zc=zc_nonic,
    )

    print("==== Sextic fields ====")
    verify_galois(
        degree=6,
        dec_types=dec_types_sextic,
        zc=zc_sextic,
        random_sample=random_sample,
    )

    print("==== Octic fields ====")
    verify_galois(
        degree=8,
        dec_types=dec_types_octic,
        zc=zc_octic,
        random_sample=random_sample,
    )

    print("==== Decic fields ====")
    verify_galois(
        degree=10,
        dec_types=dec_types_decic,
        zc=zc_decic,
    )
