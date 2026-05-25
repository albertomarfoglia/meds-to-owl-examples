import polars as pl

KEY_VARIABLES = ["Patient_ID", "Timestamp"]

HOSPITAL_STAY_VARIABLES = [
    "Length_of_Stay",
    "Admission_Mode",
    #"Discharge_Mode",
    "Visit_Type",
    "Admission_Unit",
    "Admission_Department",
    #"Discharge_Unit",
    #"Discharge_Department",
    "Emergency_Admission",
    "Number_of_Visited_Departments",
    "Visited_Units",
    "Visited_Departments",
]

PATIENT_VARIABLES = ["Gender", "Age"]

CONTEXTUAL_VARIABLES = [
    #"Rehabilitation_Discharge",
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
    "Fisher_Score",
    "WFNS_Score",
    "Hypertension",
    "Diabetes",
    "Gout",
    "Glasgow_Coma_Scale",
    "Outcome",
    "Weight",
    "Intracranial_Aneurysm_Location",
    "Intracranial_Aneurysm_Treatment_Type",
    "Unstable_Intracranial_Aneurysm",
] + HOSPITAL_STAY_VARIABLES + PATIENT_VARIABLES


SEQUENTIAL_VARIABLES = [
    "Headache",
    "Seizure",
    "Stress_Cardiomyopathy",
    "Cerebral_Salt_Wasting_Syndrome",
    "Vasospasm",
    "Hydrocephalus",
    "External_Ventricular_Drain_Details",
    "Delayed_Cerebral_Ischemia",
    "Low_Glucose",
    "High_Glucose",
    "Normal_Glucose",
    "Oxygen_Desaturation",
    "Low_BP",
    "High_BP",
    "Normal_BP",
    "Fever",
    #"Weight_Temporal",
    #"Glasgow_Coma_Scale_Temporal",
    "Nimodipine",
    "Paracetamol",
    "Norepinephrine",
    "Milrinone",
    "Morphine",
    "Antiepileptic_Treatment",
    "Antiepileptic_Treatment_SAH",
    "Diagnosis",
    "External_Ventricular_Drain",
    "Ventriculoperitoneal_Shunt",
    "Angioplasty",
    "Orotracheal_Intubation",
    "Intracranial_Aneurysm_Treatment",
    "Intraventricular_Hemorrhage",
    "Normal_Sodium",
    "High_Sodium",
    "Low_Sodium",
    "Low_Oxygen_BP",
    "Anemia",
]

SCHEMA_OVERRIDES = {
    # ------------------------------------------------------------------
    # Key variables
    # ------------------------------------------------------------------
    "Patient_ID": pl.Int64,
    "Timestamp": pl.Datetime,

    # ------------------------------------------------------------------
    # Hospital stay variables
    # ------------------------------------------------------------------
    "Length_of_Stay": pl.Float64,
    "Admission_Mode": pl.Categorical,
    "Visit_Type": pl.Categorical,
    "Admission_Unit": pl.Categorical,
    "Admission_Department": pl.Categorical,
    "Emergency_Admission": pl.Float64,
    "Number_of_Visited_Departments": pl.Float64,
    "Visited_Units": pl.Utf8,
    "Visited_Departments": pl.Utf8,

    # ------------------------------------------------------------------
    # Patient variables
    # ------------------------------------------------------------------
    "Gender": pl.Categorical,
    "Age": pl.Float64,

    # ------------------------------------------------------------------
    # Contextual variables
    # ------------------------------------------------------------------
    "Cholesterol": pl.Boolean,
    "Smoking": pl.Boolean,
    "Alcohol_Use": pl.Boolean,
    "Sleep_Apnea": pl.Boolean,
    "Anticoagulant_Therapy": pl.Boolean,
    "Antiplatelet_Therapy": pl.Boolean,

    "Hypertension_ICA": pl.Boolean,
    "Hypertension_BETA": pl.Boolean,
    "Hypertension_TZD": pl.Boolean,
    "Hypertension_ARA": pl.Boolean,
    "Hypertension_IEC": pl.Boolean,

    "Hormonal_Therapy": pl.Boolean,
    "Diabetes_Med": pl.Boolean,
    "Cholesterol_Med": pl.Boolean,

    "Myocardial_Infarction": pl.Boolean,
    "Family_History": pl.Boolean,

    "Fisher_Score": pl.Float64,
    "WFNS_Score": pl.Float64,

    "Hypertension": pl.Boolean,
    "Diabetes": pl.Boolean,
    "Gout": pl.Boolean,

    "Glasgow_Coma_Scale": pl.Float64,
    "Outcome": pl.Categorical,

    "Weight": pl.Float64,

    "Intracranial_Aneurysm_Location": pl.Categorical,
    "Intracranial_Aneurysm_Treatment_Type": pl.Categorical,
    "Unstable_Intracranial_Aneurysm": pl.Categorical,

    # ------------------------------------------------------------------
    # Sequential variables
    # ------------------------------------------------------------------
    "Headache": pl.Boolean,
    "Seizure": pl.Boolean,
    "Stress_Cardiomyopathy": pl.Boolean,
    "Cerebral_Salt_Wasting_Syndrome": pl.Boolean,
    "Vasospasm": pl.Boolean,
    "Hydrocephalus": pl.Boolean,

    "External_Ventricular_Drain_Details": pl.Utf8,

    "Delayed_Cerebral_Ischemia": pl.Boolean,

    "Low_Glucose": pl.Boolean,
    "High_Glucose": pl.Boolean,
    "Normal_Glucose": pl.Boolean,

    "Oxygen_Desaturation": pl.Boolean,

    "Low_BP": pl.Boolean,
    "High_BP": pl.Boolean,
    "Normal_BP": pl.Boolean,

    "Fever": pl.Boolean,

    "Nimodipine": pl.Boolean,
    "Paracetamol": pl.Boolean,
    "Norepinephrine": pl.Boolean,
    "Milrinone": pl.Boolean,
    "Morphine": pl.Boolean,

    "Antiepileptic_Treatment": pl.Boolean,
    "Antiepileptic_Treatment_SAH": pl.Boolean,

    "Diagnosis": pl.Categorical,

    "External_Ventricular_Drain": pl.Boolean,
    "Ventriculoperitoneal_Shunt": pl.Boolean,
    "Angioplasty": pl.Boolean,
    "Orotracheal_Intubation": pl.Boolean,
    "Intracranial_Aneurysm_Treatment": pl.Boolean,
    "Intraventricular_Hemorrhage": pl.Boolean,

    "Normal_Sodium": pl.Boolean,
    "High_Sodium": pl.Boolean,
    "Low_Sodium": pl.Boolean,

    "Low_Oxygen_BP": pl.Boolean,
    "Anemia": pl.Boolean,
}