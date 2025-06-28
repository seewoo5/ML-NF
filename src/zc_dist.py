"""
Compute distribution of square-indexed zeta coefficients for octic number fields.
"""
import pathlib
import polars as pl

from collections import defaultdict
from typing import List


current_dir = pathlib.Path(__file__).resolve().parent


def zc_G_dist(
    df: pl.DataFrame,
    indices: List[int],
    galois_group_labels: List[str],
    galois_group_names: List[str],
    ans: List[int],
) -> str:
    df_cnt = {}
    df_g_cnt = defaultdict(int)
    for g in galois_group_labels:
        df_g_cnt[g] += df.filter(pl.col("galois_label") == g).shape[0]
        for n in indices:
            for a in ans:
                df_cnt[(n, g, a)] = df.filter(
                    (pl.col("galois_label") == g) & (pl.col(f"a_{n:05d}") == a)
                ).shape[0]

    preamble = (
        "\\begin{table}[h]\n"
        "   \\begin{center}\n"
    )
    preamble += "        \\begin{tabular}{" + "c|" * (2 * len(indices)) + "c}\n"
    preamble += "            \\toprule\n"
    preamble += "            "
    for i, n in enumerate(indices):
        if i < len(indices) - 1:
            preamble += f"& \\multicolumn{{2}}{{c|}}{{$a_{{{n}}}$}} "
        else:
            preamble += f"& \\multicolumn{{2}}{{c}}{{$a_{{{n}}}$}} \\\\\n"
    preamble += "            \\midrule\n"
    preamble += "            $G$"
    for _ in indices:
        for g in galois_group_names:
            preamble += f" & ${g}$"
    preamble += " \\\\\n"
    preamble += "            \\midrule\n"
    body_zc_given_G = ""
    for a in ans:
        body_zc_given_G += f"            {a} "
        for n in indices:
            for g in galois_group_labels:
                val = df_cnt.get((n, g, a))
                if val == 0:
                    body_zc_given_G += "& - "
                else:
                    body_zc_given_G += f"& {val / df_g_cnt[g]:.2f} "
        body_zc_given_G += "\\\\\n"

    body_G_given_zc = ""
    for a in ans:
        body_G_given_zc += f"            {a} "
        for n in indices:
            val_an = sum(df_cnt.get((n, g, a)) for g in galois_group_labels)
            for g in galois_group_labels:
                val = df_cnt.get((n, g, a))
                if val == 0:
                    body_G_given_zc += "& - "
                else:
                    body_G_given_zc += f"& {val / val_an:.2f} "
        body_G_given_zc += "\\\\\n"
    end = (
        "            \\bottomrule\n"
        "        \\end{tabular}\n"
        "   \\end{center}\n"
        "\\end{table}\n"
    )
    return preamble + body_zc_given_G + end, preamble + body_G_given_zc + end


if __name__ == "__main__":
    # Sextic
    df = pl.read_csv(current_dir / "data_nf/nf_6_galois.csv", schema_overrides={f"c_{i}": pl.Int128 for i in range(6)})

    print("==== Sextic fields, p^2-th indices ====")
    sextic_zc_given_G, sextic_G_given_zc = zc_G_dist(
        df,
        indices=[2 ** 2, 3 ** 2, 5 ** 2, 7 ** 2, 17 ** 2, 19 ** 2],
        galois_group_labels=["6T1", "6T2"],
        galois_group_names=["C_{6}", "S_{3}"],
        ans=[0, 1, 3, 6, 21],
    )
    with open(current_dir / "table_tex/sextic_zc_given_G_p2.tex", "w") as f:
        f.write(sextic_zc_given_G)
    with open(current_dir / "table_tex/sextic_G_given_zc_p2.tex", "w") as f:
        f.write(sextic_G_given_zc)

    print("==== Sextic fields, p^3-th indices ====")
    sextic_zc_given_G, sextic_G_given_zc = zc_G_dist(
        df,
        indices=[2 ** 3, 3 ** 3, 5 ** 3, 7 ** 3, 17 ** 3, 19 ** 3],
        galois_group_labels=["6T1", "6T2"],
        galois_group_names=["C_{6}", "S_{3}"],
        ans=[0, 1, 2, 4, 10, 56],
    )
    with open(current_dir / "table_tex/sextic_zc_given_G_p3.tex", "w") as f:
        f.write(sextic_zc_given_G)
    with open(current_dir / "table_tex/sextic_G_given_zc_p3.tex", "w") as f:
        f.write(sextic_G_given_zc)

    # Decic
    df = pl.read_csv(current_dir / "data_nf/nf_10_galois_sq_cb_qu_100000.csv", schema_overrides={f"c_{i}": pl.Int128 for i in range(10)})
    print("==== Decic fields, p^2-th indices ====")
    decic_zc_given_G, decic_G_given_zc = zc_G_dist(
        df,
        indices=[2 ** 2, 3 ** 2, 5 ** 2, 7 ** 2, 11 ** 2, 19 ** 2],
        galois_group_labels=["10T1", "10T2"],
        galois_group_names=["C_{10}", "D_{5}"],
        ans=[0, 1, 3, 5, 15, 55],
    )
    with open(current_dir / "table_tex/decic_zc_given_G_p2.tex", "w") as f:
        f.write(decic_zc_given_G)
    with open(current_dir / "table_tex/decic_G_given_zc_p2.tex", "w") as f:
        f.write(decic_G_given_zc)

    print("==== Decic fields, p^5-th indices ====")
    decic_zc_given_G, decic_G_given_zc = zc_G_dist(
        df,
        indices=[2 ** 5, 3 ** 5, 5 ** 5, 7 ** 5],
        galois_group_labels=["10T1", "10T2"],
        galois_group_names=["C_{10}", "D_{5}"],
        ans=[0, 1, 2, 6, 126, 2002],
    )
    with open(current_dir / "table_tex/decic_zc_given_G_p5.tex", "w") as f:
        f.write(decic_zc_given_G)
    with open(current_dir / "table_tex/decic_G_given_zc_p5.tex", "w") as f:
        f.write(decic_G_given_zc)

    # Octic
    df = pl.read_csv(current_dir / "data_nf/nf_8_galois.csv", schema_overrides={f"c_{i}": pl.Int128 for i in range(8)})
    df_nonabelian = df.filter(pl.col("galois_label").is_in(["8T4", "8T5"]))

    print("==== Nonabelian octic fields, p^2-th indices ====")
    octic_zc_given_G, octic_G_given_zc = zc_G_dist(
        df_nonabelian,
        indices=[2 ** 2, 3 ** 2, 5 ** 2, 7 ** 2, 11 ** 2, 13 ** 2],
        galois_group_labels=["8T4", "8T5"],
        galois_group_names=["D_{4}", "Q_{8}"],
        ans=[0, 1, 2, 3, 4, 10, 36],
    )
    with open(current_dir / "table_tex/octic_nonab_zc_given_G_p2.tex", "w") as f:
        f.write(octic_zc_given_G)
    with open(current_dir / "table_tex/octic_nonab_G_given_zc_p2.tex", "w") as f:
        f.write(octic_G_given_zc)
