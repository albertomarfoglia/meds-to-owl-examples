from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from string import Template
import hashlib
from itertools import count


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
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

TEMPLATES = {
    "ProblemCondition": Template(
        """
<http://nvasc/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/ProblemCondition> .
<http://nvasc/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc/$event_id> <http://sphn.org/hasRecordDateTime> "$date"^^<http://www.w3.org/2001/XMLSchema#dateTime> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasCondition> <http://nvasc.org/$event_id> .
"""
    ),
    "Procedure": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Procedure> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc.org/$event_id> <http://sphn.org/hasStartDateTime> "$date"^^<http://www.w3.org/2001/XMLSchema#dateTime> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasProcedure> <http://nvasc.org/$event_id> .
"""
    ),
    "DrugAdministration": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/DrugAdministrationEvent> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasDrug> <http://nvasc.org/drug_$code> .
<http://nvasc.org/$event_id> <http://sphn.org/hasStartDateTime> "$date"^^<http://www.w3.org/2001/XMLSchema#dateTime> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasDrugAdministrationEvent> <http://nvasc.org/$event_id> .
"""
    ),
    "MeasurementCode": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Measurement> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc.org/$event_id> <http://sphn.org/hasStartDateTime> "$date"^^<http://www.w3.org/2001/XMLSchema#dateTime> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasMeasurement> <http://nvasc.org/$event_id> .
"""
    ),
    "MeasurementNumeric": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Measurement> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc.org/$event_id> <http://sphn.org/hasStartDateTime> "$date"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
<http://nvasc.org/$event_id> <http://sphn.org/hasResult> _:b$b1 .
_:b$b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/AssessmentResult> .
_:b$b1 <http://sphn.org/hasQuantity> _:b$b2 .
_:b$b2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Quantity> .
_:b$b2 <http://sphn.org/hasValue> "$numeric_value"^^<http://www.w3.org/2001/XMLSchema#float> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasMeasurement> <http://nvasc.org/$event_id> .
"""
    ),
    "Gender": Template(
        """
<http://nvasc.org/gender_$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/AdministrativeGender> .
<http://nvasc.org/gender_$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasGender> <http://nvasc.org/gender_$event_id> .
"""
    ),
    "Age": Template(
        """
<http://nvasc.org/age_$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Age> .
<http://nvasc.org/age_$event_id> <http://sphn.org/hasQuantity> _:b$b1 .
_:b$b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Quantity> .
_:b$b1 <http://sphn.org/hasValue> "$numeric_value"^^<http://www.w3.org/2001/XMLSchema#float> .
_:b$b1 <http://sphn.org/hasUnit> "years"^^<http://www.w3.org/2001/XMLSchema#string> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasAge> <http://nvasc.org/age_$event_id> .
"""
    ),
    "DiagnosisCode": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Diagnosis> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
        
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasDiagnosis> <http://nvasc.org/$event_id> .
    """
    ),
    "DiagnosisNumeric": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Diagnosis> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasQuantity> _:b$b1 .
_:b$b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Quantity> .
_:b$b1 <http://sphn.org/hasValue> "$numeric_value"^^<http://www.w3.org/2001/XMLSchema#float> .

<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasDiagnosis> <http://nvasc.org/$event_id> .
    """
    ),
}

def make_id(*parts):

    txt = "|".join([str(x) for x in parts])

    return hashlib.md5(txt.encode()).hexdigest()


blank_node_counter = count()

def new_bnode():
    return next(blank_node_counter)

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
        b1=new_bnode(),
        b2=new_bnode()
    )


def parse_code(code: str):

    if "//" in code:
        variable, value = code.split("//", 1)
    else:
        variable, value = code, None

    return variable, value
