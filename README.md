# Machine Learning Number Fields


## Requirements


Make sure that [SageMath](https://github.com/sagemath/sage) is installed. Also, we will use the following Python libraries, which will be installed in Sage shell (see **Install** below).

- [Polars](https://pola.rs/)
- [lmfdb-lite](https://github.com/roed314/lmfdb-lite)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/doc/stable/index.html)
- [tqdm](https://tqdm.github.io/)

The following libraries will be used outside of Sage (for ML experiments).

- [Polars](https://pola.rs/)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/doc/stable/index.html)


## Install


To use external python libraries in Sage, you need to install them through Sage shell:

1. Open Sage shell in terminal as

    ```
    sage --sh
    ```

2. Use `pip` or `pip3` to install `polars` and `lmfdb-lite`. If you are using Mac OS, you probably need to install [`polars-lts-cpu`](https://pypi.org/project/polars-lts-cpu/), see [this issue](https://github.com/pola-rs/polars/issues/12292).
    ```
    pip3 install polars-lts-cpu tqdm scikit-learn matplotlib numpy
    pip3 install -U "lmfdb-lite[pgbinary] @ git+https://github.com/roed314/lmfdb-lite.git"
    ```

3. Exit the Sage shell.
    ```
    exit
    ```

Now you can use these libraries in Sage as:

```
$ sage
┌────────────────────────────────────────────────────────────────────┐
│ SageMath version 10.3, Release Date: 2024-03-19 │
│ Using Python 3.11.8. Type "help()" for help. │
└────────────────────────────────────────────────────────────────────┘
sage: import polars as pl
sage: df = pl.DataFrame()
sage: df
shape: (0, 0)
┌┐
╞╡
└┘
```


## Setup virtual environment and install other packages


```
python3 -m venv .venv
source .venv/bin/activate
pip3 install polars-lts-cpu scikit-learn matplotlib numpy ipykernel
```


## Download data


```
cd src
sage nf_query.sage
```
Data will be stored in the directory `src/data_nf`. This would take about 3 hours.
