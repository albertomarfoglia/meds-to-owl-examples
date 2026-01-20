icd_codes = {
    "ICD//10//00P6X0Z": "dve",
    "ICD//10//Z98.6": "atl",
    "ICD//10//0BH17EZ": "iot",
}

atc_codes = {
    "ATC//C08CA06": "nimodipine",
    "ATC//N02BE01": "paracetamol",
    "ATC//C01CA03": "nad",
    "ATC//C01CE02": "corotrop",
    "ATC//N02AA01": "morphine",
}


NUMERIC_CODES = {
    "HOSPITAL_STAY_LENGTH//Days": "hospital_stay_length",
    "GCS_SCORE//Gcs": "gcs",
    "ACT_NUMBER//Received medical treatments": "nb_acte",
    "AGE//Years": "age",
    "outcome": "outcome",
}

CATEGORICAL_PREFIXES = {
    "GENDER//": "gender",
    "ENTRY_MODE//": "entry",
    "ENTRY_CODE//": "entry_code",
    "ICA//": "ica",
    "ICA_THERAPY//": "ica_therapy",
    "HYPERCHOLESTEROLEMIA//": "hct",
    "TTT//": "ttt",
}

BINARY_PREFIXES = {
    "FEVER//": "fever",
    "O2_CLINICAL//": "o2_clinic",
    "O2_BLOOD//": "o2",
    "HYPERTENSION//": "hta",
    "SMOKING//": "tabagisme",
    "ALCOHOL//": "etOH",
    "DIABETES//": "diabete",
    "HEADACHE//": "headache",
    "UNSTABLE_ICA//": "instable",
    "VASOSPASM//": "vasospasme",
    "IVH//": "ivh",
}

EVENTS_COLUMNS = list(icd_codes.values()) + list(atc_codes.values())

FEATURE_COLUMNS = (
    list(NUMERIC_CODES.values()) +
    list(CATEGORICAL_PREFIXES.values()) + 
    list(BINARY_PREFIXES.values()) + 
    EVENTS_COLUMNS
)
