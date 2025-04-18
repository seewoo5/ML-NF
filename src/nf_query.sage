from sage.all import *

import argparse
import json
import pathlib

import polars as pl
from lmf import db
from tqdm import tqdm


current_path = pathlib.Path(__file__).resolve().parent
nf_db = db.nf_fields


def powers(N, pows):
    ls = set()
    i = 1
    for p in pows:
        while i**p <= N:
            ls.add(i**p)
            i += 1
        i = 1
    ls = sorted(list(ls))
    return ls


def zc(poly, N, powers_only=None):
    R = PolynomialRing(ZZ, "x")
    F.<a> = NumberField(R(poly))
    zc = F.zeta_coefficients(N)
    if powers_only is not None:
        zc = [zc[i-1] for i in powers(N, powers_only)]
    return zc


def load_nf(
    degree=None,
    include_galois_gp=None,
    galois_only=False,
    N=1000,
    powers_only=None,
    limit=None,
    filename=None,
):
    """Make a polars dataframe of number fields of given degree.
    Columns are `label`, `rank`, (`galois_label`), `c_0`, ..., `c_{degree-1}`, `a_1`, `a_2`, ..., `a_N`.
    """
    filter = {}
    if degree is not None:
        filter["degree"] = degree
    if galois_only:
        # Only galois extensions
        filter["is_galois"] = True

    cols = ["label", "coeffs", "r2"]
    if include_galois_gp:
        cols.append("galois_label")

    qfs = nf_db.search(filter, cols, limit=limit)
    qfs = list(qfs)
    
    columns = ["rank"]
    if include_galois_gp:
        columns.append("galois_label")
    for i in range(degree):
        columns.append(f"c_{i}")
    if powers_only is None:
        for i in range(1, N+1):
            columns.append(f"a_{i:05d}")
    else:
        for i in powers(N, powers_only):
            columns.append(f"a_{i:05d}")

    df = None
    df_label = None
    
    chunk_size = 10000
    for i in tqdm(range(0, len(qfs), chunk_size), desc="loading data"):
        labels = []
        data = []
        for F in qfs[i:i+chunk_size]:
            label = F["label"]
            r2 = F["r2"]
            r1 = degree - 2 * r2
            r = r1 + r2 - 1
            if include_galois_gp:
                galois_label = F["galois_label"]
            labels.append(label)
            F_data = [r]
            if include_galois_gp:
                F_data.append(galois_label)
            F_data += list(float(x) for x in F["coeffs"][:-1]) + list(zc(F["coeffs"], N, powers_only))
            data.append(F_data)
        if df is None:
            df_label = pl.DataFrame(labels, schema=["label"])
            df = pl.DataFrame(data, schema=columns)
        else:
            df_label.extend(pl.DataFrame(labels, schema=["label"]))
            df.extend(pl.DataFrame(data, schema=columns))

    df = pl.concat([df_label, df], how="horizontal")

    print(f"Total number of fields: {len(df)}")
    if filename is None:
        filename = f"nf_{degree}.csv"
    else:
        filename = f"{filename}.csv"
    filepath = current_path / "data_nf" / filename
    df.write_csv(filepath)
    return df


pathlib.Path(current_path / "data_nf").mkdir(parents=True, exist_ok=True)

# Read queries from `nf_query.json` file
with open(current_path / "nf_query.json") as f:
    query = json.load(f)
    for k, v in query.items():
        print(k)
        df = load_nf(
            degree=v["degree"],
            include_galois_gp=v["include_galois_gp"],
            galois_only=v["galois_only"],
            N=v["N"],
            powers_only=v["powers_only"],
            limit=v["limit"],
            filename=f"{v['filename']}",
        )
