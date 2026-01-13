# Neurovasc on MEDS
`neurovasc_on_meds` is a Python-based workflow for converting clinical datasets into RDF graphs using the **MEDS (Medical Event Data Standard)** framework. This repository accompanies the **MEDS2RDF** library described [here](https://github.com/TeamHeKA/meds2rdf).

The workflow enables:

* Translation of clinical data into the MEDS event-based data model.
* Generation of RDF graphs compliant with the MEDS-OWL ontology.
* Semantic validation using SHACL to ensure consistency with MEDS constraints.
* Reproducible, ontology-based representation of patient care pathways.

The included example dataset focuses on neurovascular patient pathways for ruptured intracranial aneurysm, as described in:
[Jhee, J.H. et al. (2025). "Predicting Clinical Outcomes from Patient Care Pathways Represented with  Temporal Knowledge Graphs"](https://doi.org/10.1007/978-3-031-94575-5_16). 

## Repository Structure

```
neurovasc_on_meds/
├── LICENSE
├── MESSY.yaml                  # Configuration file for the MEDS-extract pipeline
├── input                       # Folder for raw or synthetic input CSVs
├── intermediate
│   └── neurovasc_codes.csv     # Preprocessed code mappings
├── main.ipynb                  # Notebook demonstrating the full workflow
├── requirements.txt            # Python dependencies
└── utils
    ├── pre_MEDS.py             # Preprocessing scripts for MEDS-extract
    └── synthetic_generator.py  # Scripts to generate synthetic neurovasc data
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/albertomarfoglia/neurovasc_on_meds.git
cd neurovasc_on_meds
```

2. (Recommended) Create a virtual environment:

```bash
# ----- Using venv -----
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# ----- Using conda -----
conda create -n venv python=3.12   # Replace "venv" and Python version as needed
conda activate venv
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

> ⚠ **Note:** `meds2rdf` requires `polars<0.20`. Using a dedicated environment avoids conflicts with other projects.

## Usage / Quick Start

This section demonstrates the full workflow decsribed in the [main.ipynb](main.ipynb), from synthetic data generation to RDF conversion and validation.

### Step 1: Prepare directories

```python
import os

os.makedirs("input", exist_ok=True)
os.makedirs("intermediate", exist_ok=True)
os.makedirs("output", exist_ok=True)
```

### Step 2: Generate synthetic dataset

```python
from utils.synthetic_generator import generate_synthetic_dataset

df_input = generate_synthetic_dataset(3, output_csv="input/syn_data.csv")
```
> The first argument (`3`) specifies the **number of patients** to generate in the synthetic dataset. You can change this number to generate more or fewer patients.

### Step 3: Preprocess data to MEDS format

```python
from utils.pre_MEDS import generate_meds_preprocessed

df_intermediate = generate_meds_preprocessed(df_input, output_path="intermediate")
```

### Step 4: Run MEDS ETL pipeline

```python
from MEDS_transforms.runner import main
import shutil

# Remove previous output
shutil.rmtree("output", ignore_errors=True)

main([
    "pkg://MEDS_extract.configs._extract.yaml",
    "--overrides",
    "input_dir=intermediate",
    "output_dir=output",
    "event_conversion_config_fp=MESSY.yaml",
    "dataset.name=Neurovasc",
    "dataset.version=1.0",
])
```

This produces a **MEDS-compliant dataset** in `output/`.

### Step 5: Convert MEDS dataset to RDF

```python
from meds2rdf import MedsRDFConverter
import os

converter = MedsRDFConverter("output")
graph = converter.convert(include_splits=True)

OUTPUT_DIR = "output/rdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

graph.serialize(destination=f"{OUTPUT_DIR}/output_dataset.ttl", format="turtle")
graph.serialize(destination=f"{OUTPUT_DIR}/output_dataset.xml", format="xml")
graph.serialize(destination=f"{OUTPUT_DIR}/output_dataset.nt", format="nt")

print("Conversion complete! RDF files saved in output/rdf/")
```

### Step 6: Validate RDF with SHACL

```python
from pyshacl import validate

data_graph = "output/rdf/output_dataset.ttl"
shacl_graph = "https://raw.githubusercontent.com/TeamHeKA/meds-ontology/refs/heads/main/shacl/meds-shapes.ttl"

conforms, results_graph, results_text = validate(
    data_graph, shacl_graph=shacl_graph, advanced=True
)

print(results_text)
```

This ensures the RDF graph conforms to **MEDS-OWL constraints**.

## References

* [MEDS2RDF Python Library](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL Ontology](https://github.com/TeamHeKA/meds-ontology)
* [Jhee, J.H. et al. (2025). "Predicting Clinical Outcomes from Patient Care Pathways Represented with  Temporal Knowledge Graphs"](https://doi.org/10.1007/978-3-031-94575-5_16).

## License

This project is licensed under the [LICENSE](LICENSE) file.

## Acknowledgments

This work builds on the **MEDS** framework and the **Semantic Web** standards (RDF/OWL, SHACL) for biomedical data integration.
