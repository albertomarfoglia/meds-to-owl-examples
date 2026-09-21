import shap
import numpy as np
import pandas as pd
import re

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_best_features(
    output_dir,
    explainer,
    feature_names,
    X,
    classes=None,  # kept for API compatibility
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X)

    if not isinstance(shap_values, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray from SHAP, got {type(shap_values)}"
        )

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected SHAP values with shape "
            f"(n_samples, n_features), got {shap_values.shape}"
        )

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
    )

    plt.savefig(
        output_dir / "shap_summary.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


from pathlib import Path

import numpy as np
import pandas as pd


def save_top_shap_features_per_class(
    explainer,
    output_dir,
    X,
    feature_names,
    classes,
    quantile=0.95,
):
    """
    Select features according to the quantile of mean absolute SHAP
    importance.

    For the current binary XGBoost model, SHAP returns a single
    (n_samples, n_features) matrix rather than separate matrices
    for TRUE and FALSE.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X)

    if not isinstance(shap_values, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray from SHAP, got {type(shap_values)}"
        )

    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected SHAP values with shape "
            f"(n_samples, n_features), got {shap_values.shape}"
        )

    feature_names = np.array(
        [
            f.replace("_count", "").replace("//", "_")
            for f in feature_names
        ]
    )

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"SHAP has {shap_values.shape[1]} features, "
            f"but feature_names contains {len(feature_names)} names."
        )

    # One importance value per feature
    importance = np.abs(shap_values).mean(axis=0)

    threshold = np.quantile(importance, quantile)

    selected_idx = np.where(importance >= threshold)[0]

    sorted_idx = selected_idx[
        np.argsort(importance[selected_idx])[::-1]
    ]

    sorted_features = feature_names[sorted_idx]
    sorted_importance = importance[sorted_idx]

    # The current SHAP output is not class-specific.
    # Store the result under the positive class name if that is
    # the class represented by the model output.
    class_name = classes[0] if classes else "TRUE"

    top_features = {
        class_name: {
            "features": sorted_features,
            "importance": sorted_importance,
            "threshold": threshold,
        }
    }

    rows = []

    for rank, (feature, value) in enumerate(
        zip(sorted_features, sorted_importance),
        start=1,
    ):
        rows.append(
            {
                "class": class_name,
                "rank": rank,
                "feature": feature,
                "importance": value,
                "threshold": threshold,
            }
        )

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / f"shap_features_quantile_{quantile}.csv",
        index=False,
    )

    np.save(
        output_dir / f"features_{quantile}.npy",
        df["feature"].unique(),
    )

    return top_features


def sanitize_for_uri(feature_name: str) -> str:
    s = " ".join(
        feature_name.replace("\\", "\\\\")
        .replace("\r\n", "\n")
        .replace("\t", "\n")
        .replace("\r", "\n")
        .replace('"', "")
        .split()
    )

    unsafe_chars = r'[<>"{}|\\^`\[\]\s]'

    s = re.sub(unsafe_chars, " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def clean_xgb_feature_name(n):
    n = str(n)
    n = re.sub(r"[\[\]<>]", "", n)  # remove forbidden chars
    #n = re.sub(r"[^0-9a-zA-Z_]+", "_", n)  # replace other specials
    # n = re.sub(r"_+", "_", n)              # collapse multiple _
    n = n.replace("//", "_")
    n = n.strip("_")
    return n