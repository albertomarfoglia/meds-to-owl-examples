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
        "Admission_Mode",
        # "Discharge_Mode",
        "Visit_Type",
        "Admission_Unit",
        "Admission_Department",
        # "Discharge_Unit",
        # "Discharge_Department",
        "Emergency_Admission",
        "Cholesterol",
        "Smoking",
        "Alcohol_Use",
        "Sleep_Apnea",
        "Anticoagulant_Therapy",
        "Antiplatelet_Therapy",
        "Hypertension_ICA",
        "Hypertension_BETA",
        "Hypertension_TZD",
        "Hypertension_ARA",
        "Hypertension_IEC",
        "Hormonal_Therapy",
        "Diabetes_Med",
        "Cholesterol_Med",
        "Myocardial_Infarction",
        "Family_History",
        "Hypertension",
        "Diabetes",
        "Gout",
        "Intracranial_Aneurysm_Location",
        "Unstable_Intracranial_Aneurysm",
        "External_Ventricular_Drain_Details",
        # "Rehabilitation_Discharge",
    ],

    "DiagnosisNumeric": [
        "Number_of_Visited_Departments",
        "Length_of_Stay",
        "Glasgow_Coma_Scale",
        "Fisher_Score",
        "WFNS_Score",
        "Weight",
    ],

    "ProblemCondition": [
        "Stress_Cardiomyopathy",
        "Cerebral_Salt_Wasting_Syndrome",
        "Vasospasm",
        "Hydrocephalus",
        "Intraventricular_Hemorrhage",
        "Delayed_Cerebral_Ischemia",
    ],

    "DrugAdministration": [
        "Nimodipine",
        "Paracetamol",
        "Norepinephrine",
        "Milrinone",
        "Morphine",
        "Antiepileptic_Treatment",
        "Antiepileptic_Treatment_SAH",
    ],

    "Procedure": [
        "Diagnosis",
        "External_Ventricular_Drain",
        "Ventriculoperitoneal_Shunt",
        "Angioplasty",
        "Orotracheal_Intubation",
        "Intracranial_Aneurysm_Treatment",
        "Intracranial_Aneurysm_Treatment_Type"
    ],

    "Gender": ["Gender"],

    "Age": ["Age"],

    "MeasurementCode": [
        "Headache",
        "Seizure",
        "Low_Glucose",
        "High_Glucose",
        "Normal_Glucose",
        "Oxygen_Desaturation",
        "Low_BP",
        "High_BP",
        "Normal_BP",
        "Fever",
        "Normal_Sodium",
        "High_Sodium",
        "Low_Sodium",
        "Low_Oxygen_BP",
        "Anemia",
    ],

    # "MeasurementNumeric": [
    #     "Glasgow_Coma_Scale_Temporal",
    #     "Weight",
    # ],
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
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasCondition> <http://nvasc.org/$event_id> .
$date_triple
"""
    ),
    "Procedure": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Procedure> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasProcedure> <http://nvasc.org/$event_id> .
$date_triple
"""
    ),
    "DrugAdministration": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/DrugAdministrationEvent> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasDrug> <http://nvasc.org/drug_$code> .
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasDrugAdministrationEvent> <http://nvasc.org/$event_id> .
$date_triple
"""
    ),
    "MeasurementCode": Template(
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Measurement> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasMeasurement> <http://nvasc.org/$event_id> .
$date_triple
"""
    ),
    "MeasurementNumeric": Template(
# <http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Measurement> .
# <http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
# <http://nvasc.org/$event_id> <http://sphn.org/hasCode> <http://nvasc.org/code_$code> .
# <http://nvasc.org/$event_id> <http://sphn.org/hasResult> _:b$b1 .
# _:b$b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/AssessmentResult> .
# _:b$b1 <http://sphn.org/hasQuantity> _:b$b2 .
# _:b$b2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Quantity> .
# _:b$b2 <http://sphn.org/hasValue> "$numeric_value"^^<http://www.w3.org/2001/XMLSchema#float> .
# <http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasMeasurement> <http://nvasc.org/$event_id> .
# $date_triple
        """
<http://nvasc.org/$event_id> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Diagnosis> .
<http://nvasc.org/$event_id> <http://www.w3.org/2000/01/rdf-schema/label> "$label"^^<http://www.w3.org/2001/XMLSchema#string> .
<http://nvasc.org/$event_id> <http://sphn.org/hasQuantity> _:b$b1 .
_:b$b1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://sphn.org/Quantity> .
_:b$b1 <http://sphn.org/hasValue> "$numeric_value"^^<http://www.w3.org/2001/XMLSchema#float> .
<http://nvasc.org/synth_patient_$patient_id> <http://nvasc.org/hasDiagnosis> <http://nvasc.org/$event_id> .
$date_triple
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
$date_triple
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
$date_triple
    """
    ),
}

def make_id(*parts):

    txt = "|".join([str(x) for x in parts])

    return hashlib.md5(txt.encode()).hexdigest()


blank_node_counter = count()

def new_bnode():
    return next(blank_node_counter)

DATE_PREDICATES = {
    "ProblemCondition": "hasRecordDateTime",
    "Procedure": "hasStartDateTime",
    "DrugAdministration": "hasStartDateTime",
    "MeasurementCode": "hasStartDateTime",
    "MeasurementNumeric": "hasStartDateTime",
    "DiagnosisNumeric": "hasRecordDateTime",
    "DiagnosisCode": "hasRecordDateTime",
}

def render_event(event: SemanticEvent):

    template = TEMPLATES[event.template_name]

    date_triple = ""
    if event.time and event.template_name in DATE_PREDICATES:
        predicate = DATE_PREDICATES[event.template_name]
        date_triple = f'''<http://nvasc.org/{event.event_id}> <http://sphn.org/{predicate}> "{event.time.isoformat()}"^^<http://www.w3.org/2001/XMLSchema#dateTime> .'''

    return template.substitute(
        patient_id=event.patient_id,
        event_id=event.event_id,
        code=event.code.replace("//", "_").replace(" ", "-"),
        label=event.label,
        date_triple=date_triple,
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
