# patient_feature_queries.py
#
# This module defines documented SPARQL queries for extracting
# patient-level clinical features from a MEDS-style RDF event graph.
#
# Assumptions:
# - Each clinical observation is represented as a meds:Event
# - Each Event has:
#     - meds:hasSubject (required)
#     - meds:hasCode (required)
#     - meds:time (optional)
#     - meds:numericValue (optional)
# - Clinical meaning is encoded in meds:codeString values such as:
#     - FEVER_0, FEVER_1
#     - GCS_SCORE_Gcs
#     - IVH_1
#     - HOSPITAL_STAY_LENGTH_Days

PREFIXES = """
PREFIX meds: <https://teamheka.github.io/meds-ontology#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


# ---------------------------------------------------------------------
# FEVER FEATURES
# ---------------------------------------------------------------------

# Query: Patients with FEVER = 0
#
# Meaning:
# Returns patients who explicitly have a fever code indicating absence
# (FEVER_0).
#
# Use case:
# Binary patient feature: has_fever = 0
FEVER_ABSENT = PREFIXES + """
SELECT DISTINCT ?subject
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code .
  ?code meds:codeString "FEVER//0"^^xsd:string .
}
"""


# Query: Patients with FEVER = 1
#
# Meaning:
# Returns patients who explicitly have fever present.
#
# Use case:
# Binary patient feature: has_fever = 1
FEVER_PRESENT = PREFIXES + """
SELECT DISTINCT ?subject
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code .
  ?code meds:codeString "FEVER//1"^^xsd:string .
}
"""


# ---------------------------------------------------------------------
# GCS FEATURES
# ---------------------------------------------------------------------

# Query: All GCS measurements per patient
#
# Meaning:
# Returns all recorded GCS values for each patient.
#
# Use case:
# Time series analysis or quality checks.
GCS_ALL_VALUES = PREFIXES + """
SELECT ?subject ?gcs ?time
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code ;
         meds:numericValue ?gcs .
  ?code meds:codeString "GCS_SCORE//Gcs"^^xsd:string .
  OPTIONAL { ?event meds:time ?time }
}
ORDER BY ?subject ?time
"""


# Query: Minimum (worst) GCS per patient
#
# Meaning:
# Computes the minimum GCS value observed for each patient.
#
# Use case:
# Severity feature in outcome prediction models.
GCS_MIN_PER_PATIENT = PREFIXES + """
SELECT ?subject (MIN(?gcs) AS ?minGCS)
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code ;
         meds:numericValue ?gcs .
  ?code meds:codeString "GCS_SCORE//Gcs"^^xsd:string .
}
GROUP BY ?subject
"""


# Query: Maximum (best) GCS per patient
#
# Meaning:
# Computes the maximum GCS value observed for each patient.
#
# Use case:
# Recovery or improvement indicator.
GCS_MAX_PER_PATIENT = PREFIXES + """
SELECT ?subject (MAX(?gcs) AS ?maxGCS)
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code ;
         meds:numericValue ?gcs .
  ?code meds:codeString "GCS_SCORE//Gcs"^^xsd:string .
}
GROUP BY ?subject
"""


# ---------------------------------------------------------------------
# DIAGNOSIS / CONDITION FEATURES
# ---------------------------------------------------------------------

# Query: Patients with intraventricular hemorrhage (IVH)
#
# Meaning:
# Returns patients with IVH explicitly present.
#
# Use case:
# Binary neurological condition feature.
IVH_PRESENT = PREFIXES + """
SELECT DISTINCT ?subject
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code .
  ?code meds:codeString "IVH//1"^^xsd:string .
}
"""


# Query: Patients without IVH (closed-world assumption)
#
# Meaning:
# Returns patients for whom IVH_1 is not recorded.
#
# Warning:
# Assumes absence of evidence = evidence of absence.
IVH_ABSENT = PREFIXES + """
SELECT DISTINCT ?subject
WHERE {
  ?subject a meds:Subject .
  FILTER NOT EXISTS {
    ?event a meds:Event ;
           meds:hasSubject ?subject ;
           meds:hasCode ?code .
    ?code meds:codeString "IVH//1"^^xsd:string .
  }
}
"""


# ---------------------------------------------------------------------
# TREATMENT / MEDICATION FEATURES
# ---------------------------------------------------------------------

# Query: Patients treated with Nimodipine
#
# Meaning:
# Identifies patients receiving Nimodipine (ATC:C08CA06),
# commonly used for cerebral vasospasm.
#
# Use case:
# Treatment exposure feature.
NIMODIPINE_TREATED = PREFIXES + """
SELECT DISTINCT ?subject
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code .
  ?code meds:codeString "ATC//C08CA06"^^xsd:string .
}
"""


# ---------------------------------------------------------------------
# HOSPITALIZATION FEATURES
# ---------------------------------------------------------------------

# Query: Hospital length of stay per patient
#
# Meaning:
# Extracts length of hospital stay (in days) for each patient.
#
# Assumption:
# One LOS event per patient.
HOSPITAL_STAY_LENGTH = PREFIXES + """
SELECT ?subject ?los
WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code ;
         meds:numericValue ?los .
  ?code meds:codeString "HOSPITAL_STAY_LENGTH//Days"^^xsd:string .
}
"""


# ---------------------------------------------------------------------
# WIDE PATIENT FEATURE MATRIX
# ---------------------------------------------------------------------
PATIENT_FEATURE_MATRIX = PREFIXES + """
SELECT
  (STRAFTER(STR(?subject), "subject/") AS ?sub_id)

  (MAX(?los)         AS ?hospital_stay_length)
  (MIN(?gcs)         AS ?gcs)
  (MAX(?nb_acte)     AS ?nb_acte)
  (MAX(?gender)      AS ?gender)
  (MAX(?entry)       AS ?entry)
  (MAX(?entry_code)  AS ?entry_code)
  (MAX(?ica)         AS ?ica)
  (MAX(?ica_therapy) AS ?ica_therapy)
  (MAX(?ttt)         AS ?ttt)
  (MAX(?hct)         AS ?hct)

  (MAX(?age)         AS ?age)
  (MAX(?outcome)     AS ?outcome)

  (MAX(IF(?cs = "FEVER//1", 1, 0))              AS ?fever)
  (MAX(IF(?cs = "O2_CLINICAL//1", 1, 0))        AS ?o2_clinic)
  (MAX(IF(?cs = "O2_BLOOD//1", 1, 0))           AS ?o2)
  (MAX(IF(?cs = "HYPERTENSION//1", 1, 0))       AS ?hta)
  (MAX(IF(?cs = "SMOKING//1", 1, 0))            AS ?tabagisme)
  (MAX(IF(?cs = "ALCOHOL//1", 1, 0))            AS ?etOH)
  (MAX(IF(?cs = "DIABETES//1", 1, 0))           AS ?diabete)
  (MAX(IF(?cs = "HEADACHE//1", 1, 0))           AS ?headache)
  (MAX(IF(?cs = "UNSTABLE_ICA//1", 1, 0))       AS ?instable)
  (MAX(IF(?cs = "VASOSPASM//1", 1, 0))          AS ?vasospasme)
  (MAX(IF(?cs = "IVH//1", 1, 0))                AS ?ivh)

  (MAX(IF(?cs = "ICD//10//00P6X0Z", 1, 0))      AS ?dve)
  (MAX(IF(?cs = "ICD//10//Z98.6", 1, 0))        AS ?atl)
  (MAX(IF(?cs = "ICD//10//0BH17EZ", 1, 0))      AS ?iot)
  
  (MAX(IF(?cs = "ATC//C08CA06", 1, 0))      AS ?nimodipine)
  (MAX(IF(?cs = "ATC//N02BE01", 1, 0))      AS ?paracetamol)
  (MAX(IF(?cs = "ATC//C01CA03", 1, 0))      AS ?nad)
  (MAX(IF(?cs = "ATC//C01CE02", 1, 0))      AS ?corotrop)
  (MAX(IF(?cs = "ATC//N02AA01", 1, 0))      AS ?morphine)

WHERE {
  ?event a meds:Event ;
         meds:hasSubject ?subject ;
         meds:hasCode ?code .
  ?code meds:codeString ?cs .

  OPTIONAL {
    ?event meds:numericValue ?los .
    FILTER(?cs = "HOSPITAL_STAY_LENGTH//Days")
  }

  OPTIONAL {
    ?event meds:numericValue ?gcs .
    FILTER(?cs = "GCS_SCORE//Gcs")
  }

  OPTIONAL {
    ?event meds:numericValue ?nb_acte .
    FILTER(?cs = "ACT_NUMBER//Received medical treatments")
  }

  OPTIONAL {
    ?event meds:numericValue ?age .
    FILTER(?cs = "AGE//Years")
  }

  OPTIONAL {
    ?event meds:numericValue ?outcome .
    FILTER(?cs = "outcome")
  }

  BIND(xsd:integer(REPLACE(?cs, "^HYPERCHOLESTEROLEMIA//", "")) AS ?hct).
  BIND(xsd:integer(REPLACE(?cs, "^ENTRY_MODE//", "")) AS ?entry).
  BIND(xsd:integer(REPLACE(?cs, "^ENTRY_CODE//", "")) AS ?entry_code).
  BIND(xsd:integer(REPLACE(?cs, "^GENDER//", "")) AS ?gender).
  BIND(xsd:integer(REPLACE(?cs, "^ICA//", "")) AS ?ica).
  BIND(xsd:integer(REPLACE(?cs, "^ICA_THERAPY//", "")) AS ?ica_therapy).
  BIND(xsd:integer(REPLACE(?cs, "^TTT//", "")) AS ?ttt).
}
GROUP BY ?subject
"""


# ---------------------------------------------------------------------
# QUERY REGISTRY
# ---------------------------------------------------------------------

# Dictionary collecting all queries for programmatic access
ALL_QUERIES = {
    "fever_absent": FEVER_ABSENT,
    "fever_present": FEVER_PRESENT,
    "gcs_all_values": GCS_ALL_VALUES,
    "gcs_min": GCS_MIN_PER_PATIENT,
    "gcs_max": GCS_MAX_PER_PATIENT,
    "ivh_present": IVH_PRESENT,
    "ivh_absent": IVH_ABSENT,
    "nimodipine": NIMODIPINE_TREATED,
    "hospital_stay_length": HOSPITAL_STAY_LENGTH,
    "feature_matrix": PATIENT_FEATURE_MATRIX,
}

from rdflib import Graph 

def run_query(graph: Graph, query_name: str, query: str) -> list:
    results = graph.query(query)
    
    # Handle the case where results.vars is None
    vars_ = [str(v) for v in results.vars] if results.vars is not None else []

    # Build rows
    return [{v: row[v] for v in vars_} for row in results]