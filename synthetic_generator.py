"""
Synthetic Dataset Generator (with remote defaults)
-------------------------------------------------
This version automatically downloads correlation and transition
probability files from GitHub when needed and caches them locally.

Usage:
    from synthetic_generator import generate_synthetic_dataset
    df = generate_synthetic_dataset(500)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm, genextreme, exponweib
from itertools import accumulate
import joblib
import math
from sklearn.preprocessing import MinMaxScaler
from typing import List
import requests


# ------------------------------------------------------------
# Remote defaults
# ------------------------------------------------------------

DEFAULT_CORR_URL = (
    "https://github.com/TeamHeKA/neurovasc/raw/refs/heads/main/data/nantes_correlations.joblib"
)

DEFAULT_TRANSITION_URL = (
    "https://raw.githubusercontent.com/TeamHeKA/neurovasc/refs/heads/main/data/care_transitions_probs.csv"
)

CACHE_DIR = os.path.expanduser("~/.cache/synthgen")
os.makedirs(CACHE_DIR, exist_ok=True)


# ------------------------------------------------------------
# Utilities: Download + Cache
# ------------------------------------------------------------

def _cached_download(url: str) -> str:
    """
    Downloads a remote file into ~/.cache/synthgen/ if not already cached.
    Returns the local file path.
    """
    filename = os.path.join(CACHE_DIR, os.path.basename(url))

    if not os.path.exists(filename):
        print(f"Downloading {url} → {filename}")
        resp = requests.get(url)
        resp.raise_for_status()

        with open(filename, "wb") as f:
            f.write(resp.content)

    return filename

# ------------------------------------------------------------
# Main API
# ------------------------------------------------------------

def generate_synthetic_dataset(
    n_patients: int,
    correlation_path: str = DEFAULT_CORR_URL,
    transition_path: str = DEFAULT_TRANSITION_URL,
    output_csv: str | None = None
) -> pd.DataFrame:
    """
    Generate synthetic neurovascular dataset.

    Parameters
    ----------
    n_patients : int
        Number of synthetic patients.
    correlation_path : str
        URL or local path for correlation matrix.
    transition_path : str
        URL or local path for transition probabilities.
    output_csv : str, optional
        If provided, saves dataset to disk.

    Returns
    -------
    pd.DataFrame
    """

    
    # ------------------------------------------------------------
    # Load remote or local files
    # ------------------------------------------------------------
    if correlation_path.startswith("http"):
        correlation_path = _cached_download(correlation_path)
    if transition_path.startswith("http"):
        transition_path = _cached_download(transition_path)

    correlation_matrix = joblib.load(correlation_path)
    transitions = pd.read_csv(transition_path, index_col=0)

    # List of varibles
    numerical = ['hospital_stay_length', 'gcs', 'nb_acte', 'age']
    categorical = ['gender', 'entry', 'entry_code', 'ica', 'ttt', 'ica_therapy', 'fever', 'o2_clinic', 'o2', 'hta', 'hct', 'tabagisme', 'etOH', 'diabete', 'headache', 'instable', 'vasospasme', 'ivh', 'outcome']
    events = ['nimodipine',  'paracetamol', 'nad', 'corotrop', 'morphine', 'dve', 'atl', 'iot']
    events_end = events + ['finish']
    y = ['back2home', 'reabilitation', 'death'] # outcome
    y_probs = [0.443396, 0.432075, 0.124529] # outcome probability from the distribution

    # Generate variables based on the statistics
    df = pd.DataFrame({
        'hospital_stay_length': map(round, genextreme.rvs(-0.4091639605356321, 13.2154345852118, 13.507892218123956, n_patients)),
        'gcs': map(round, norm.rvs(14.866037735849057, 1.079385463913648, n_patients)),
        'nb_acte': map(round, exponweib.rvs(1.7487636231551846, 0.7992842590334144, 0.9388125774311487, 22.6608165193314, n_patients)),
        
        'gender': np.random.choice(['F', 'M'], size=n_patients, p=[0.615094, 0.384906]),
        'entry': np.random.choice(['7', '6', '3', '13', '2', '8', '0', '1', '5'], size=n_patients, p=[0.289412, 0.254118, 0.157647, 0.145882, 0.103529, 0.023529, 0.018824, 0.004706, 0.002353]),    
        'entry_code': np.random.choice(['3850', '2083', '1215', '3412', '2071', '3810', '2072', '3851', '3811', '5042', '3830', '2082', '2073', '3762', '3411', '1214', '2086', '3577', '1224', '1151', '2611', '1412', '2612', '1314', '1211', '3770', '2011', '5014', '3760'], size=n_patients, p=[0.501887, 0.192453, 0.084906, 0.033962, 0.030189, 0.020755, 0.018868, 0.016981, 0.013208, 0.011321, 0.011321, 0.009434, 0.009434, 0.007547, 0.00566, 0.003774, 0.003774, 0.003774, 0.001887, 0.001887, 0.001887, 0.001887, 0.001887, 0.001887, 0.001886, 0.001886, 0.001886, 0.001886, 0.001886]),
        'ica': np.random.choice(['ACoA', 'ACM', 'ACI', 'ACoP', 'ACA', 'PICA', 'TB', 'V', 'hyperdebit', 'ACP', 'AChoA', 'Dissection', 'ACerebS', 'BA', 'AICA', 'TN', 'Aucun', "ACoAde_l'artère_communicante_antérieur", 'ACL', 'JA'], size=n_patients, p=[0.309859, 0.205634, 0.171831, 0.073239, 0.067606, 0.056338, 0.053521, 0.011268, 0.008451, 0.005634, 0.005634, 0.005634, 0.005634, 0.002817, 0.002817, 0.002817, 0.002817, 0.002817, 0.002816, 0.002816]),
        'ttt': np.random.choice(['spire', 'remodeling', 'clip', 'web', 'flow_diverter'], size=n_patients, p=[0.933962, 0.024528, 0.020755, 0.018868, 0.001887]),
        'ica_therapy': np.random.choice(['0', 'loxen', 'amlodipine', 'nicardipin', 'lercanidipine', 'amlor', 'lercan', 'exforge', 'axeler'], size=n_patients, p=[0.966038, 0.007547, 0.007547, 0.00566, 0.003774, 0.003774, 0.001887, 0.001887, 0.001886]),

        'fever': np.random.choice(['0', 'fever'], size=n_patients, p=[0.898113, 0.101887]),
        'o2_clinic': np.random.choice(['0', 'low'], size=n_patients, p=[0.813208, 0.186792]),
        'o2': np.random.choice(['0', 'low'], size=n_patients, p=[0.722642, 0.277358]),
        'hta': np.random.choice(['0', '1'], size=n_patients, p=[0.935849, 0.064151]),
        'hct': np.random.choice(['0', '1', 'hypercholester'], size=n_patients, p=[0.949057, 0.049057, 0.001886]),
        'tabagisme': np.random.choice(['0', '1'], size=n_patients, p=[0.864151, 0.135849]),
        'etOH': np.random.choice(['0', '1'], size=n_patients, p=[0.958491, 0.041509]),
        'diabete': np.random.choice(['0', '1'], size=n_patients, p=[0.958491, 0.041509]),
        'headache': np.random.choice(['1', '0'], size=n_patients, p=[0.835849, 0.164151]),
        'instable': np.random.choice(['0', '1'], size=n_patients, p=[0.917625, 0.082375]),
        'vasospasme': np.random.choice(['1', '0'], size=n_patients, p=[0.984906, 0.015094]),
        'ivh': np.random.choice(['0', '1'], size=n_patients, p=[0.932075, 0.067925]),
        'age': map(round, genextreme.rvs(0.27689720964297965, 51.599845037531225, 14.34488206435922, n_patients)),
        'outcome': np.random.choice(y, size=n_patients, p=y_probs)
        })
    df[categorical] = df[categorical].apply(lambda x : pd.factorize(x)[0])
    df.head()

    # Cholesky decomposition to introduce correlations
    L = np.linalg.cholesky(correlation_matrix)
    synthetic_data = df @ L
    synthetic_data.columns = df.columns

    def round_with_prob(x, n_cats):
        dec, ent = math.modf(x)
        ent = int(ent)
        return np.random.choice([ent, ent+1], size=1, p=[1-dec, dec])[0] % n_cats

    def cut(x):
        dec, ent = math.modf(x)
        ent = int(ent)
        return ent

    for feature in ['gender', 'fever', 'o2_clinic', 'o2', 'hta', 'tabagisme', 'etOH', 'diabete', 'headache', 'instable', 'vasospasme', 'ivh']:
        synthetic_data[[feature]] = MinMaxScaler(feature_range=(0,1.99)).fit_transform(synthetic_data[[feature]])
        synthetic_data[feature] = synthetic_data[feature].apply(lambda x: cut(x))

    for feature, n_classes in zip(['entry', 'entry_code', 'ica', 'ttt', 'ica_therapy', 'hct'], [9, 29, 20, 5, 9, 3]):
        synthetic_data[[feature]] = MinMaxScaler(feature_range=(0, n_classes-0.01)).fit_transform(synthetic_data[[feature]])
        synthetic_data[feature] = synthetic_data[feature].apply(lambda x: cut(x))

    synthetic_data.head()


    # Generate outcomes
    arr = synthetic_data['outcome']
    sorted_indices = np.argsort(arr)
    transformed_array = np.zeros_like(arr)
    q1 = int(n_patients*y_probs[0])
    q2 = int(n_patients*(y_probs[0]+y_probs[1]))
    transformed_array[sorted_indices[:q1]] = 0
    transformed_array[sorted_indices[q1:q2]] = 1
    transformed_array[sorted_indices[q2:]] = 2

    synthetic_data['outcome'] = transformed_array


    #  Add care pathway events
    real_correlation = synthetic_data.corr().to_numpy()
    start_probs = [0.47381546, 0.09476309, 0.00997506, 0, 0.00997506, 0.24189526, 0.00249377, 0.16708229, 0]

    def generate_care_path():
        event = np.random.choice(events_end, size=1, p=start_probs)[0]
        path = [event]

        while event != 'finish':
            event = np.random.choice(events_end, size=1, p=transitions[event].values)[0]
            if event in path:
                event = 'finish'
            path += [event]
        
        return path

    def generate_times_path(path):
        path = path[:-1]
        indv_times = map(round, norm.rvs(24, 5, len(path)))
        acc_times = list(accumulate(indv_times))

        sol = [-1] * len(events)
        for i, e in enumerate(path):
            sol[events.index(e)] = acc_times[i]
        
        return sol

    df_events = pd.DataFrame([generate_times_path(generate_care_path()) for _ in range(n_patients)], columns=events)

    # Final synthetic tabular data 
    full = pd.concat([synthetic_data, df_events], axis=1)

    if output_csv:
        full.to_csv(output_csv, index=False)

    return full