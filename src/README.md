## Running ML experiments

### Classifying Galois groups of Galois extensions

1. Make sure you have queried the required datasets using `nf_query.sage`.

2. Run

    - `galois_group_4_9.ipynb`
    - `galois_group_6_10.ipynb`
    - `galois_group_8.ipynb`

    Each file runs ML experiments, saves the relevant figures under `figs`, and generate LaTeX code for decision tree models under `tree_tex`.
    It may take about an hour for each notebook.


### Compute distribution of zeta coefficients

`zc_dist.py` computes the distribution of zeta coefficients.
Run
```
python zc_dist.py
```
It computes

1. $\mathbb{P}[a_n(K) = a | G]$ for given $n, G$ and possible $a$ (`zc_given_G`)
2. $\mathbb{P}[G | a_n(K) = a]$ for given $n, a$ and possible $G$ (`G_given_zc`)

and generate the corresponding LaTeX tables under the directory `table_tex`.
$n$ are chosen as powers of primes.

### Verify decomposition types

For each degree and prime $p$, `verify_galois.sage` checks possible decomposition types of $p$ for various fields with given degree.
Run
```
sage verify_galois.sage
```