import json
from typing import List, Literal, Optional

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def primes(n: int) -> List[int]:
    out = list()
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p] and sieve[p] % 2 == 1):
            out.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return out


def powers(N: int, pows: List[Optional[int]]) -> List[int]:
    ls = set()
    i = 1
    for p in pows:
        while i**p <= N:
            ls.add(i**p)
            i += 1
        i = 1
    ls = sorted(list(ls))
    return ls


def df_stats(df: pl.DataFrame, label: Literal["rank", "galois_label"] = "rank"):
    """
    Print the statistics of the dataframe
    """
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Head: {df.head()}")

    print(f"Label: {label}")
    df_label = df.group_by(label).len().with_columns((pl.col("len") / pl.sum("len")).alias("percent")).sort(label)
    print(f"Label value counts: {df_label}")


def zeta_count(
    df: pl.DataFrame,
    n: int,
    label_type: Optional[Literal["rank", "galois_label"]] = None,
    label: Optional[str] = None,
):
    """
    Give a statistic of a_n(K) for a given index n.
    """
    if label_type is not None:
        assert label is not None, "If label_type is given, label must be given too."
        df = df.filter(pl.col(label_type) == label)

    print(f"Index: {n}")
    df_n = df.group_by(
        f"a_{n:05d}"
    ).len().with_columns((pl.col("len") / pl.sum("len")).alias("percent")).sort(f"a_{n:05d}")
    print(df_n)


def X_y(
    df: pl.DataFrame,
    degree: int = 2,
    feature_type: Literal["c", "a", "a_p"] = "c",
    label: Literal["rank", "galois_label"] = "rank",
    N: int = 1000,
    powers_only: Optional[List[int]] = None,
):
    if feature_type == "c":
        columns_ = [f"c_{i}" for i in range(degree)]
    elif feature_type == "a":
        if powers_only is None:
            columns_ = [f"a_{i:05d}" for i in range(1, N+1)]
        else:
            columns_ = [f"a_{i:05d}" for i in powers(N, powers_only)]

    elif feature_type == "a_p":
        # primes only
        columns_ = [f"a_{i:05d}" for i in primes(N)]

    X = df.select(columns_)
    y = df.select(label)
    return X, y


def run_experiment(
    df: pl.DataFrame,
    name: str,
    test_size: float = 0.2,
    feature_type: str = Literal["c", "a", "a_p"],
    degree: int = 2,
    label: str = "rank",
    model_type: Literal["dt", "rf", "lr"] = "dt",
    num_coeffs: Optional[int] = 1000,
    powers_only: Optional[List[int]] = None,
    lr_max_iter: int = 10000,
    lr_solver: str = "lbfgs",
):
    X, y = X_y(df, degree=degree, feature_type=feature_type, label=label, N=num_coeffs, powers_only=powers_only)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print(f"Data: {name}, {feature_type}")
    print(f"Train: {X_train.shape}")
    print(f"Test ({label}): {X_test.shape}")

    # Train the model
    if model_type == "dt":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "lr":
        model = LogisticRegression(random_state=42, max_iter=lr_max_iter, solver=lr_solver)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    clr = classification_report(y_test, y_pred, digits=4)
    print(clr)
    print("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp = disp.plot(cmap=plt.cm.Blues)
    if model_type == "dt":
        plt.savefig(f"figs/dt/{name}_cm.png", dpi=1200)
    elif model_type == "lr":
        plt.savefig(f"figs/lr/{name}_cm.png", dpi=1200)
    plt.show()
    return model, clr


def run_experiments(
    df: pl.DataFrame,
    experiments_file: str,
    class_names: List[str],
    current_dir: str,
    lr_po: Optional[List[int]] = None,
    save_tree_fig: bool = True,
):
    with open(f"{current_dir}/experiments/{experiments_file}.json", "r") as f:
        experiments = json.load(f)

    if "rank" in experiments_file:
        label = "rank"
    elif "galois" in experiments_file:
        label = "galois_label"
    else:
        raise ValueError(f"Unknown label type in {experiments_file}")

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        deg = exp.get("degree")
        nc = exp.get("num_coeffs")
        ft = exp.get("feature_type")
        mt = exp.get("model_type")
        po = exp.get("powers_only")
        model, _ = run_experiment(
            df,
            name=exp.get("name"),
            test_size=0.2,
            feature_type=ft,
            degree=deg,
            label=label,
            model_type=exp.get("model_type"),
            num_coeffs=nc,
            powers_only=po,
            lr_max_iter=exp.get("lr_max_iter", 10000),
            lr_solver=exp.get("lr_solver", "lbfgs"),
        )

        if mt == "dt" and save_tree_fig:
            if ft == "a":
                if po is None:
                    feature_names = [f"a_{i:05d}" for i in range(1, nc+1)]
                else:
                    feature_names = [f"a_{i:05d}" for i in powers(nc, po)]
            elif ft == "a_p":
                feature_names = [f"a_{i:05d}" for i in primes(nc)]
            elif ft == "c":
                feature_names = [f"c_{i}" for i in range(deg)]

            plot_tree(
                model,
                filled=True,
                feature_names=feature_names,
                class_names=class_names,
                impurity=False,
            )
            plt.savefig(current_dir / f"figs/dt/{exp['name']}.png", dpi=1200)
            plt.show()
        elif mt == "lr":
            if label == "galois_label" and ft == "a" and po is None:
                # Check distribution of whole and special-power coefficients
                k = min(30, nc - 1)
                po_indices = powers(nc, lr_po)
                po_indices = [i - 1 for i in po_indices]  # -1 because of 0-indexing
                po_name = []
                if 2 in lr_po:
                    po_name.append("sq")
                if 3 in lr_po:
                    po_name.append("cb")
                if 5 in lr_po:
                    po_name.append("qu")
                po_name = "_".join(po_name)
                lr_coefficient_dist(model, k=k, save_hist=f"{current_dir}/figs/lr/{exp['name']}_whole.png")
                lr_coefficient_dist(model, indices=po_indices, save_hist=f"{current_dir}/figs/lr/{exp['name']}_{po_name}.png")
            elif label == "rank" and ft == "c":
                print(f"model weights: {model.coef_}")
                print(f"model bias: {model.intercept_}")



def print_tree_structure(clf: DecisionTreeClassifier, max_depth: Optional[int] = None):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if max_depth is None or node_depth[i] <= max_depth:
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node with value={value}.".format(
                        space=node_depth[i] * "  ", node=i, value=np.around(values[i], 3)
                    )
                )
            else:
                print(
                    "{space}node={node} is a split node with value={value}: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.".format(
                        space=node_depth[i] * "  ",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i],
                        value=np.around(values[i], 3),
                    )
                )


def lr_coefficient_dist(
    model: LogisticRegression,
    indices: Optional[List[int]] = None,
    k: Optional[int] = None,
    bins: int = 10,
    save_hist: Optional[str] = None,
):
    """
    Get a distribution of the coefficients of a logistic regression model
    """
    coeff = model.coef_[0]
    if indices is not None:
        coeff = coeff[indices]
    coeff_abs = np.abs(coeff)

    # Basic statistics
    print(f"Mean: {np.mean(coeff_abs)}")
    print(f"Median: {np.median(coeff_abs)}")
    print(f"Std: {np.std(coeff_abs)}")

    if k is not None:
        top_k = np.partition(coeff, -k)[-k:]
        top_k = np.sort(top_k)[::-1]

        top_k_indices = np.argpartition(coeff, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(coeff[top_k_indices])[::-1]]
        print(f"Top {k}: {top_k}")
        print(f"Top {k} indices: {top_k_indices}")

        bottom_k = np.partition(coeff, k)[:k]
        bottom_k = np.sort(bottom_k)
        bottom_k_indices = np.argpartition(coeff, k)[:k]
        bottom_k_indices = bottom_k_indices[np.argsort(coeff[bottom_k_indices])]
        print(f"Bottom {k}: {bottom_k}")
        print(f"Bottom {k} indices: {bottom_k_indices}")

    # Histogram of the abolute values of the weights
    plt.hist(coeff_abs, bins=bins)
    plt.xlabel("|Model weight|")
    plt.ylabel("Frequency")
    plt.title("Distribution of absolute value of model weights")
    if save_hist is not None:
        plt.savefig(save_hist)
    plt.show()


def check_value_in(df: pl.DataFrame, index: int, values: List):
    col_values = df[f"a_{index:05d}"].unique().to_list()
    assert set(col_values) <= set(values), f"Invalid values in column a_{index:05d}: {col_values} not in {values}"
