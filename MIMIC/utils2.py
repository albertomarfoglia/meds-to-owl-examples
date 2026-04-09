import polars as pl


def generate_samples(n_folds: int, labels: pl.LazyFrame, size: int, seed: int | None, fixed_true = False):
    true_values = (
        labels.filter(pl.col("boolean_value"))
        # .sort(pl.col("prediction_time"), descending=False)
        # .group_by("subject_id")
        # .head(1)
        .group_by("subject_id")
        .agg(pl.all().sample(n=1, seed=seed))
        .explode(pl.all().exclude("subject_id"))
        .collect(engine="streaming")
    )

    false_values = (
        labels.filter(~pl.col("boolean_value"))
        .join(
            true_values.select("subject_id").lazy(),
            on="subject_id",
            how="anti",
        )
        # .sort(pl.col("prediction_time"), descending=False)
        # .group_by("subject_id")
        # .head(1)
        .group_by("subject_id")
        .agg(pl.all().sample(n=1, seed=seed))
        .explode(pl.all().exclude("subject_id"))
        .collect(engine="streaming")
    )

    fold_length = int(size / 2)

    false_values = false_values.sample(n=fold_length * n_folds, shuffle=True, seed=seed, with_replacement=False)
    false_samples = [
        false_values.slice(i * fold_length, fold_length) for i in range(n_folds)
    ]

    if fixed_true:
        true_values = true_values.sample(n=fold_length, shuffle=True, seed=seed, with_replacement=False)
        return [false_samples[i].extend(true_values) for i in range(n_folds)]
    else:
        true_values = true_values.sample(n=fold_length * n_folds, shuffle=True, seed=seed, with_replacement=False)

        true_samples = [
            true_values.slice(i * fold_length, fold_length) for i in range(n_folds)
        ]

        return [false_samples[i].extend(true_samples[i]) for i in range(n_folds)]


bad_strings = ["___", "None", "N/A", "", " "]

remove_long_text = (
    pl.when(
        (pl.col("text_value").is_in(bad_strings))
        | (pl.col("text_value").str.len_chars() > 50)
    )
    .then(None)
    .otherwise(pl.col("text_value"))
    .alias("text_value")
)

birthdate_to_age = [
    (
        pl.when(pl.col("code") == "MEDS_BIRTH")
        .then((pl.col("prediction_time") - pl.col("time")) / pl.duration(days=365))
        .otherwise(None)
        .alias("numeric_value")
    ),
    pl.when(pl.col("code") == "MEDS_BIRTH")
    .then(None)
    .otherwise(pl.col("time"))
    .alias("time"),
]

before_prediction_time = (pl.col("time").is_null()) | (
    pl.col("time") <= pl.col("prediction_time")
)

swap_text_with_numeric = ({
    "numeric_value": pl.coalesce(pl.col("parsed"), pl.col("numeric_value")),
    "text_value": pl.when(pl.col("parsed").is_not_null()).then(None).otherwise(pl.col("text_value")),
})

parse_txt_to_float = pl.col("text_value").cast(pl.Float64, strict=False)

regenerate_ids = (
    pl.col("subject_id")
    .cast(pl.Utf8)
    .cast(pl.Categorical)
    .to_physical()
    .alias("subject_id")
)

# no_vitals = ~pl.col("code").str.contains_any(vitals)

is_bp = pl.col("code") == "Blood Pressure"

# quick sanity check: contains a slash + strict-looking numeric pattern "num/num"
bp_text_valid = pl.col("text_value").is_not_null() & pl.col("text_value").str.contains(
    r"^\s*\d+(\.\d+)?/\d+(\.\d+)?\s*$"
)

# produce a list: for Blood Pressure valid texts -> [high, low] (floats, sorted desc)
# otherwise -> single-element list [numeric_value] so explode() yields one row for non-BP rows
bp_values_list = (
    pl.when(is_bp & bp_text_valid)
    .then(
        pl.col("text_value")
        .str.strip_chars()
        .str.split("/")
        .list.eval(
            pl.element().cast(pl.Float64, strict=False)
        )  # cast both parts to float
        .list.sort(descending=True)  # higher first
    )
    .otherwise(pl.concat_list([pl.col("numeric_value")]))
)


to_systolic_and_dyastolic_pressure = (
    pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
    .then(
        pl.when(
            pl.col("numeric_value")
            == pl.col("bp_values").max().over(["subject_id", "time", "orig_text"])
        )
        .then(pl.lit("Systolic blood pressure"))
        .otherwise(pl.lit("Diastolic blood pressure"))
    )
    .otherwise(pl.col("code"))
)


set_BP_values_text_to_null = (
    pl.when(pl.col("orig_code").str.contains("Blood Pressure"))
    .then(pl.lit(None).cast(pl.Utf8))
    .otherwise(pl.col("text_value"))
)

derived_columns = ["bp_values", "orig_code", "orig_text", "parsed"]


def extract_labels_array(outcomes: pl.DataFrame):
    labels = (outcomes.select(["subject_id", "boolean_value"])
        .group_by("subject_id")
        .agg(pl.col("boolean_value").first().cast(pl.Int8).alias("boolean_value"))
        .sort("subject_id")
        .select("boolean_value")
        .to_series()
        .to_list()
    )
    return labels
