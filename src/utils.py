import json
import re
from typing import List, Literal, Optional

from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
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


# Code generated using o4-mini-high
def numbers_with_power_multiples(N, pows):
    """
    Return all integers 1 < x ≤ N such that in the prime‐factorization
    of x = ∏ p_i^e_i, each exponent e_i satisfies (e_i % k == 0) for at least
    one k ∈ pows.

    For example, if pows = [2,3], then 144 = 2^4 * 3^2 is allowed (since 4, 2 are multiples of 2).

    Steps:
      1. Find all primes p up to floor(N ** (1/min(pows))).  Any prime bigger
         than that cannot have p^(min(pows)) ≤ N, so it is irrelevant.
      2. For each of those “relevant” primes, build a list of allowed powers [1, p^e, …]
         where e runs from 1 upward, and we include p^e exactly when e % k == 0 for some k.
      3. Do a depth-first recursion over those primes, choosing for each prime one of its
         “allowed powers” (including 1, which means “use p^0”), multiply them together,
         and collect every result ≤ N.
    """

    if N < 2 or not pows:
        return []

    # Step 0: Sort pows and find the minimum (so we know how far we need to sieve).
    pows = sorted(pows)
    min_exp = pows[0]

    # Step 1: Only primes p ≤ N**(1/min_exp) can contribute:
    from math import floor
    cutoff = int(floor(N ** (1.0 / min_exp)))
    relevant_primes = primes(cutoff)

    # Step 2: Precompute for each relevant prime p, a list of “allowed powers” ≤ N.
    #   We always include 1 (meaning “don’t use this prime”), then for e=1,2,3,…, we keep
    #   multiplying until p^e > N.  Whenever e % k == 0 for at least one k∈pows,
    #   we append p^e.
    prime_powers = []
    for p in relevant_primes:
        powers = [1]  # exponent 0 is always “allowed” (just skip the prime)
        val = p
        e = 1
        while val <= N:
            # if e is a multiple of any entry in pows, keep p^e
            if any(e % k == 0 for k in pows):
                powers.append(val)
            e += 1
            val *= p
        prime_powers.append(powers)

    results = set()

    def dfs(idx, current):
        # if we’ve decided on a power for every relevant prime, record it (if >1)
        if idx == len(prime_powers):
            if current > 1:
                results.add(current)
            return

        for power_val in prime_powers[idx]:
            new_val = current * power_val
            if new_val > N:
                # increasing power_val (or later primes) will only make it bigger
                continue
            dfs(idx + 1, new_val)

    # Kick off recursion with idx=0 and current=1
    dfs(0, 1)
    return sorted(results)


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
            columns_ = [f"a_{i:05d}" for i in numbers_with_power_multiples(N, powers_only)]

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
        md = exp.get("max_depth")
        cln = exp.get("class_names")
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
                    feature_names = [f"a_{i:05d}" for i in numbers_with_power_multiples(nc, po)]
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

            # Save tex file
            latex = generate_forest_latex_from_tree(model, class_names=cln, max_depth=None)
            with open(current_dir / f"tree_tex/{exp['name']}.tex", "w") as f:
                f.write(latex)
            if md is not None:
                latex_md = generate_forest_latex_from_tree(model, class_names=cln, max_depth=md)
                with open(current_dir / f"tree_tex/{exp['name']}_depth_{md}.tex", "w") as f:
                    f.write(latex_md)
                

        elif mt == "lr":
            if label == "galois_label" and ft == "a" and po is None:
                # Check distribution of whole and special-power coefficients
                k = min(30, nc - 1)
                po_indices = numbers_with_power_multiples(nc, lr_po)
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


# Code generated using o4-mini-high
def generate_forest_latex_from_tree(model, class_names=None, max_depth=10):
    """
    Generates LaTeX code for 'forest' directly from a fitted DecisionTreeClassifier.
    Infers feature names from model.feature_names_in_ and class names from model.classes_.
    If max_depth is provided, nodes at that depth (even if not leaves) are printed as pseudo-leaves.
    Returns a string containing the forest LaTeX block.
    """
    # Helper to wrap subscripts: "a_30" -> "a_{30}"
    def wrap_subscripts(name):
        return re.sub(r"_(\w+)", r"_{\1}", name)

    feature_names = [wrap_subscripts(str(f)) for f in model.feature_names_in_]
    if class_names is None:
        class_names = [str(c) for c in model.classes_]
    tree = model.tree_

    def node_to_latex(node_id, indent=2, edge_label=None, depth=0):
        space = ' ' * indent
        edge_opt = f", {edge_label}" if edge_label is not None else ""
        is_leaf = tree.feature[node_id] == _tree.TREE_UNDEFINED
        # If reached max_depth and not a true leaf, print pseudo-leaf
        if (max_depth is not None) and (depth >= max_depth) and not is_leaf:
            return (f"{space}[{{${{\\cdots}}$}}, rectangle, draw{edge_opt}, tier=bottom]")
        if is_leaf:
            values = tree.value[node_id][0]
            class_idx = int(values.argmax())
            class_label = class_names[class_idx]
            return (f"{space}[{{${class_label}$}}, rectangle, thick, draw{edge_opt}, "
                    "tier=bottom, line width=1.5pt]")
        # split node
        feat_idx = tree.feature[node_id]
        thresh = tree.threshold[node_id]
        feat_name = feature_names[feat_idx]
        cond = f"${feat_name} \\le {thresh}$"
        node_opt = f"{{{cond}}}, draw{edge_opt}"
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        left_latex = node_to_latex(left_id, indent + 2, edge_label='label L=Y', depth=depth+1)
        right_latex = node_to_latex(right_id, indent + 2, edge_label='label R=N', depth=depth+1)
        return (f"{space}[{node_opt}\n"
                f"{left_latex}\n"
                f"{right_latex}\n"
                f"{space}]")

    preamble = ("\\begin{forest}\n"
                "  label L/.style={edge label={node[midway,left,font=\\scriptsize]{#1}}},\n"
                "  label R/.style={edge label={node[midway,right,font=\\scriptsize]{#1}}},\n"
                "  for tree={forked edge, child anchor=north, for descendants={edge=->}}\n")
    body = node_to_latex(0, indent=2, depth=0)
    end = "\\end{forest}"
    return f"{preamble}{body}\n{end}"
