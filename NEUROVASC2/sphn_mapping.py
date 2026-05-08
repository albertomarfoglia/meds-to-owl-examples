from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from string import Template
import hashlib


@dataclass
class SemanticEvent:
    patient_id: int
    variable: str
    numeric_value: Optional[float]
    time: Optional[datetime]
    template_name: str
    code: str
    label: str
    unit: Optional[str] = None
    event_id: Optional[str] = None


_SEMANTIC_MAP = {
    "DiagnosisCode": [
        "IN_MODE",
        "OUT_MODE",
        "VISIT_TYPE",
        "IN_UNIT",
        "SERVICE_ENTREE",
        "OUT_UNIT",
        "SERVICE_SORTIE",
        "EMERGENCY",
        "cholesterol",
        "tabac",
        "alcool",
        "apnee_sommeil",
        "anticoagulant",
        "antiagregant",
        "HTA_ICA",
        "HTA_BETA",
        "HTA_TZD",
        "HTA_ARA",
        "HTA_IEC",
        "hormonal_med",
        "diabete_med",
        "cholesterol_med",
        "infarctus",
        "ATCD_familiaux",
        "HTA",
        "diabete",
        "maladie_goutte",
        "SORTIE_REEDUC",
    ],
    "DiagnosisNumeric": [
        "NB_SERVICES_PARCOURUS",
        "DUREE",
        "score_glasgow",
        "score_fisher",
        "score_wfns",
    ],
    "ProblemCondition": [
        "cardiomyopathie_stress",
        "syndrome_perte_sel",
        "vasospasme",
        "hydrocephalie",
        "hemorragie_intra_vent",
        "ischemie_cerebrale_retardee",
    ],
    "DrugAdministration": [
        "nimodipine",
        "paracetamol",
        "noradrenaline",
        "milrinone",
        "morphine",
        "antiepideptique",
        "antiepideptique_HSA",
    ],
    "Procedure": [
        "diagnostic",
        "DVE",
        "DVP",
        "angioplastie",
        "intubation_orotracheale",
        "traitement_AIC",
    ],
    "Gender": ["GENDER"],
    "Age": ["AGE"],
    "MeasurementCode": [
        "cephale",
        "crise",
        "glucose_bas",
        "glucose_eleve",
        "glucose_normale",
        "desaturation_O2",
        "PA_bas",
        "PA_eleve",
        "PA_normal",
        "fievre",
        "NA_normal",
        "NA_eleve",
        "NA_bas",
        "PA_O2_bas",
        "anemie",
    ],
    "MeasurementNumeric": ["GLASGOW_SEQ", "POIDS"],
}

SEMANTIC_MAP = {value: key for key, values in _SEMANTIC_MAP.items() for value in values}


PREFIX = """
@prefix sphn: <http://sphn.org/> .
@prefix nvasc: <http://nvasc.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

TEMPLATES = {
    "ProblemCondition": Template(
        """
nvasc:$event_id a sphn:ProblemCondition ;
    rdfs:label "$label"^^xsd:string ;
    sphn:hasCode <nvasc:code_$code> ;
    sphn:hasRecordDateTime "$date"^^xsd:dateTime .

nvasc:synth_patient_$patient_id
    nvasc:hasCondition nvasc:$event_id .
"""
    ),
    "Procedure": Template(
        """
nvasc:$event_id a sphn:Procedure ;
    rdfs:label "$label"^^xsd:string ;
    sphn:hasCode nvasc:code_$code ;
    sphn:hasStartDateTime "$date"^^xsd:dateTime .

nvasc:synth_patient_$patient_id
    nvasc:hasProcedure nvasc:$event_id .
"""
    ),
    "DrugAdministration": Template(
        """
nvasc:$event_id a sphn:DrugAdministrationEvent ;
    rdfs:label "$label"^^xsd:string ;
    sphn:hasDrug nvasc:drug_$code ;
    sphn:hasStartDateTime "$date"^^xsd:dateTime .

nvasc:synth_patient_$patient_id
    nvasc:hasDrugAdministrationEvent nvasc:$event_id .
"""
    ),
    "MeasurementCode": Template(
        """
nvasc:$event_id a sphn:Measurement ;
    rdfs:label "$label"^^xsd:string ;
    sphn:hasCode nvasc:code_$code ;
    sphn:hasStartDateTime "$date"^^xsd:dateTime .

nvasc:synth_patient_$patient_id
    nvasc:hasMeasurement nvasc:$event_id .
"""
    ),
    "MeasurementNumeric": Template(
        """
nvasc:$event_id a sphn:Measurement ;
    rdfs:label "$label"^^xsd:string ;
    sphn:hasCode nvasc:code_$code ;
    sphn:hasStartDateTime "$date"^^xsd:dateTime ;
    sphn:hasResult [
        rdf:type sphn:AssessmentResult ;
        sphn:hasQuantity [
            rdf:type sphn:Quantity ;
            sphn:hasValue "$numeric_value" ;
        ]
    ] .

nvasc:synth_patient_$patient_id
    nvasc:hasMeasurement nvasc:$event_id .
"""
    ),
    "Gender": Template(
        """
nvasc:gender_$event_id a sphn:AdministrativeGender ;
    sphn:hasCode nvasc:code_$code .

nvasc:synth_patient_$patient_id
    nvasc:hasGender nvasc:gender_$event_id .
"""
    ),
    "Age": Template(
        """
nvasc:age_$event_id a sphn:Age ;
    sphn:hasQuantity [
        rdf:type sphn:Quantity ;
        sphn:hasValue "$numeric_value" ;
        sphn:hasUnit "years"
    ] .

nvasc:synth_patient_$patient_id
    nvasc:hasAge nvasc:age_$event_id .
"""
    ),
    "DiagnosisCode": Template(
        """
    nvasc:$event_id a sphn:Diagnosis ;
        rdfs:label "$label"^^xsd:string ;
        sphn:hasCode nvasc:code_$code .
        
    nvasc:synth_patient_$patient_id nvasc:hasDiagnosis nvasc:$event_id .
    """
    ),
    "DiagnosisNumeric": Template(
        """
    nvasc:$event_id a sphn:Diagnosis ;
        rdfs:label "$label" ;
        sphn:hasQuantity [ rdf:type sphn:Quantity ;
                            sphn:hasValue "$numeric_value" ] .
    
    nvasc:synth_patient_$patient_id nvasc:hasDiagnosis nvasc:$event_id .
    """
    ),
}


def make_id(*parts):

    txt = "|".join([str(x) for x in parts])

    return hashlib.md5(txt.encode()).hexdigest()


def render_event(event: SemanticEvent):

    template = TEMPLATES[event.template_name]

    return template.substitute(
        patient_id=event.patient_id,
        event_id=event.event_id,
        code=event.code.replace("//", "_"),
        label=event.label,
        date=event.time.isoformat() if event.time else "",
        numeric_value=event.numeric_value,
        unit=event.unit or "",
    )


def parse_code(code: str):

    if "//" in code:
        variable, value = code.split("//", 1)
    else:
        variable, value = code, None

    return variable, value
