import polars as pl
import numpy as np
from pathlib import Path

def process_codes(input_parquet: str, output_dir: str, prefix_map: dict):
    """
    Extract codes for each ontology, strip prefix, flatten, and save efficiently.

    Args:
        input_parquet: path to the Parquet file containing 'parent_codes' column.
        output_dir: folder to save processed arrays.
        prefix_map: dict of {prefix_name: prefix_string}, e.g. "ICD10PCS": "ICD10PCS/"
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Load Parquet lazily
    df = pl.scan_parquet(input_parquet).select(["parent_codes"])

    for name, prefix in prefix_map.items():
        print(f"Processing {name}...")

        # Lazy filter: keep lists that contain at least one code with prefix
        codes = (
            df.explode("parent_codes")
            .filter(pl.col("parent_codes").str.starts_with(prefix))
            .select(
                [
                    pl.col("parent_codes")
                    .str.replace(prefix, "", literal=True)
                    .alias("code")
                ]
            )
        )

        # Collect to memory and save as .npy
        codes_array = codes.collect()["code"].to_numpy()
        if len(codes_array) > 0:
            np.save(output_dir_path / f"{name}_codes.npy", codes_array)
            print(
                f"Saved {len(codes_array)} codes for {name} → {(name + '_codes.npy')}"
            )


PREFIX_MAP = {
    "ICD10PCS": "ICD10PCS/",
    "ICD10CM": "ICD10CM/",
    "LOINC": "LNC/",
    "RXNORM": "RXNORM/",
    "ICD9CM": "ICD9CM/",
    "SNOMED": "SNOMED/",
}

def get_proportions(outcomes: pl.LazyFrame) -> pl.LazyFrame:
    return (
        outcomes.group_by("boolean_value")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .select("boolean_value", "proportion")
    )

def sample(
    outcomes: pl.LazyFrame,
    sample_size: int = 100,
    proportions: dict | None = None,
    seed: int | None = None,
) -> pl.LazyFrame:

    if proportions is None:
        proportions_lf = get_proportions(outcomes)
    else:
        proportions_lf = pl.LazyFrame(
            {
                "boolean_value": list(proportions.keys()),
                "proportion": list(proportions.values()),
            }
        )

    # compute per-class sample sizes
    sample_sizes = proportions_lf.with_columns(
        (pl.col("proportion") * sample_size).floor().cast(pl.Int64).alias("n")
    ).select("boolean_value", "n")

    return (
        outcomes.join(sample_sizes, on="boolean_value")
        .with_columns(
            pl.int_range(0, pl.len())
            .shuffle(seed)
            .over("boolean_value")
            .alias("rand_rank")
        )
        .filter(pl.col("rand_rank") < pl.col("n"))
        .drop(["n", "rand_rank"])
    )

def get_labels(events: pl.DataFrame) -> list[int]:
    return (
        events.group_by("subject_id")
        .agg(pl.col("boolean_value").first().cast(pl.Int8).alias("boolean_value"))
        .sort("subject_id")
        .select("boolean_value")
        .to_series()
        .to_list()
        #.collect()
        #.to_series()
        #.to_list()
    )


# def sample2(
#     outcomes: pl.LazyFrame, sample_size=100, proportions=None, seed: int | None = None
# ):
#     if proportions is None:
#         proportions = get_proportions(outcomes)

#     sample_sizes = {k: int(v * sample_size) for k, v in proportions.items()}

#     sampled_frames = []
#     for cls, n in sample_sizes.items():
#         sampled = outcomes.filter(pl.col("boolean_value") == cls).sample(
#             n=n, with_replacement=False, seed=seed
#         )
#         sampled_frames.append(sampled)

#     return pl.concat(sampled_frames)


# def get_proportions2(outcomes: pl.LazyFrame):
#     counts = outcomes.group_by("boolean_value").agg(pl.len()).to_dicts()
#     total = sum(c["len"] for c in counts)
#     return {d["boolean_value"]: d["len"] / total for d in counts}
