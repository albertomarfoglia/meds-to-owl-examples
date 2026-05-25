# MEDS to OWL examples

This repository provides reference workflows for converting clinical datasets into RDF graphs using the **MEDS (Medical Event Data Standard)** framework. This repository accompanies the **MEDS2RDF** library described [here](https://github.com/TeamHeKA/meds2rdf) and the MEDS-OWL ontology documentted [here](https://teamheka.github.io/meds-ontology/).

The workflow enables:

* Translation of clinical data into the MEDS event-based data model.
* Generation of RDF graphs compliant with the MEDS-OWL ontology.
* Optional semantic validation using SHACL to ensure consistency with MEDS constraints.

The repository currently includes two dataset conversions:

* **NEUROVASC** (synthetic neurovascular care pathways): described in [Jhee, J.H. et al. (2025)](https://doi.org/10.1007/978-3-031-94575-5_16).
* **MIMIC-IV Demo on MEDS** (real-world critical care data): publishied in https://physionet.org/content/mimic-iv-demo-meds/0.0.1/.


## Repository Structure

```
meds-to-owl-examples
├── LICENSE
├── README.md
├── requirements.txt            # Install by `pip install -r`
│
├── MIMIC
│   ├── experiments.yaml        # Experiment configuration
│   ├── main.ipynb              # End-to-end MIMIC pipeline notebook
│   ├── run_aces.sh             # ACEs pipeline for cohorts generation
│   ├── run_meds_etl.sh         # MEDS extraction / ETL pipeline
│   │
│   ├── tasks                   # Predictive tasks definitions (YAML)
│   │   ├── first_24_in_icu_mortality.yaml
│   │   ├── first_48_in_hospital_los.yaml
│   │   ├── readmission_30d.yaml
│   │   ├── post_hospital_discharge_mortality_30d.yaml
│   │   └── ... (additional ICU/hospital outcome tasks)
│   │
│   └── utils
│
├── NEUROVASC
│   ├── MESSY.yaml              # MEDS-extract mapping template
│   ├── main.ipynb              # Neurovascular pipeline notebook
│   ├── pre_MEDS
│   │   └── neurovasc_codes.csv # Ontologies codes imported during meds conversion
│   └── utils
│
├── NEUROVASC2
    ├── MESSY.yaml              # UMEDS-extract mapping template
    ├── main.ipynb              # Neurovascular pipeline notebook
    └── utils
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/albertomarfoglia/meds-to-owl-examples.git
cd meds-to-owl-examples
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

## References

* [MEDS2RDF Python Library](https://github.com/TeamHeKA/meds2rdf)
* [MEDS-OWL Ontology](https://github.com/TeamHeKA/meds-ontology)
* [Jhee, J.H. et al. (2025). "Predicting Clinical Outcomes from Patient Care Pathways Represented with  Temporal Knowledge Graphs"](https://doi.org/10.1007/978-3-031-94575-5_16).

## License

This project is licensed under the [LICENSE](LICENSE) file.

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{marfoglia2026clinicaldatagoesmeds,
      title={Clinical Data Goes MEDS? Let's OWL make sense of it}, 
      author={Alberto Marfoglia and Jong Ho Jhee and Adrien Coulet},
      year={2026},
      eprint={2601.04164},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.04164}, 
}
```

## Acknowledgments

This work builds on the **MEDS** framework and the **Semantic Web** standards (RDF/OWL, SHACL) for biomedical data integration.
