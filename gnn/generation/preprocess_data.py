from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

def preprocess_sphn_kg(node_df, entity, time_opt, num_patients):
    # Get literals.
    numeric_df = node_df[node_df['r'] == '<http://sphn.org/hasValue>'].copy()
    numeric_values = pd.to_numeric(numeric_df.t.values)
    numeric_df['numeric'] = numeric_values
    numeric_arr = np.zeros((len(entity), 1))
    for i, v in numeric_df.t.items():
        num_id = entity[entity.entity == v].id
        numeric_arr[num_id] = numeric_df.numeric.loc[i]

    if time_opt == 'NT':
        np.save(f"processed_data/sphn_pc_NT_numeric_{num_patients}.npy", numeric_arr)
        print("Literals NT saved.")
    elif time_opt == 'TR':
        np.save(f"processed_data/sphn_pc_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TR saved.")
    elif time_opt == 'TS':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        times = []
        for i, t in time_df.sec.items():
            time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
            times.append(time.total_seconds())
        time_df['sec'] = times
            
        qt = QuantileTransformer(n_quantiles=10, random_state=0)
        qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
        time_df['sec'] = list(qt_times.reshape(-1,))
        for i, v in time_df.t.items():
            num_id = entity[entity.entity == v].id
            numeric_arr[num_id] = time_df.sec.loc[i]
        
        np.save(f"processed_data/sphn_pc_TS_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS saved.")
    elif time_opt == 'TS_TR':
        time_df = node_df[node_df['r'].str.contains('<http://sphn.org/hasStartDateTime>|<http://sphn.org/hasDeterminationDateTime>')].copy()
        time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>')
        times = []
        for i, t in time_df.sec.items():
            time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
            times.append(time.total_seconds())
        time_df['sec'] = times
        
        qt = QuantileTransformer(n_quantiles=10, random_state=0)
        qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
        time_df['sec'] = list(qt_times.reshape(-1,))
        for i, v in time_df.t.items():
            num_id = entity[entity.entity == v].id
            numeric_arr[num_id] = time_df.sec.loc[i]
        np.save(f"processed_data/sphn_pc_TS_TR_numeric_{num_patients}.npy", numeric_arr)
        print("Literals TS TR saved.")

# def preprocess_meds_kg(node_df, entity, time_opt, num_patients, prefix="meds"):
#     MEDS_NAMESPACE = "https://teamheka.github.io/meds-ontology#"
#     # Get literals.
#     numeric_df = node_df[node_df['r'] == f'<{MEDS_NAMESPACE}numericValue>'].copy() # can be improved using Graph
#     numeric_values = pd.to_numeric(numeric_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#double>').values)
#     numeric_df['numeric'] = numeric_values
#     numeric_arr = np.zeros((len(entity), 1))
#     for i, v in numeric_df.t.items():
#         num_id = entity[entity.entity == v].id
#         numeric_arr[num_id] = numeric_df.numeric.loc[i]

#     if time_opt == 'TS':
#         time_df = node_df[node_df['r'].str.contains(f'<{MEDS_NAMESPACE}time>')].copy() # can be improved using Graph
#         time_df['sec'] = time_df.t.str.removesuffix('^^<http://www.w3.org/2001/XMLSchema#dateTime>') # can be improved using Graph
#         times = []
#         for i, t in time_df.sec.items():
#             time = datetime.strptime(t, '%Y-%m-%dT%H:%M:%S') - datetime(2020,1,1)
#             times.append(time.total_seconds())
#         time_df['sec'] = times
            
#         print("Running quantile transfromation")
#         qt = QuantileTransformer(n_quantiles=10, random_state=0)
#         qt_times = qt.fit_transform(time_df.sec.values.reshape(-1,1))
#         time_df['sec'] = list(qt_times.reshape(-1,))
#         for i, v in time_df.t.items():
#             num_id = entity[entity.entity == v].id
#             numeric_arr[num_id] = time_df.sec.loc[i]

#     np.save(f"processed_data/{prefix}_{time_opt}_numeric_{num_patients}.npy", numeric_arr)
#     print("Literals saved.")
    
def preprocess_meds_kg(node_df, entity, time_opt, num_patients, prefix="meds"):
    MEDS_NAMESPACE = "https://teamheka.github.io/meds-ontology#"

    entity_to_id = dict(zip(entity.entity.values, entity.id.values))
    numeric_arr = np.zeros((len(entity), 1))

    # -------- Numeric literals --------
    numeric_df = node_df.loc[
        node_df['r'] == f'<{MEDS_NAMESPACE}numericValue>',
        ['t']
    ].copy()

    numeric_df['numeric'] = (
        numeric_df['t']
        .str.replace('^^<http://www.w3.org/2001/XMLSchema#double>', '', regex=False)
        .astype(float)
        .round(2)
    )

    ids = numeric_df['t'].map(entity_to_id).values
    numeric_arr[ids] = numeric_df['numeric'].values.reshape(-1, 1)

    # -------- Time literals --------
    if time_opt == 'TS':
        time_df = node_df.loc[
            node_df['r'].str.contains(f'<{MEDS_NAMESPACE}time>'),
            ['t']
        ].copy()

        time_df['sec'] = pd.to_datetime(
            time_df['t'].str.replace(
                '^^<http://www.w3.org/2001/XMLSchema#dateTime>',
                '',
                regex=False
            )
        )

        time_df['sec'] = (
            time_df['sec'] - pd.Timestamp("2020-01-01")
        ).dt.total_seconds()

        from sklearn.preprocessing import RobustScaler
        time_df['sec'] = RobustScaler().fit_transform(time_df[['sec']]).ravel()

        ids = time_df['t'].map(entity_to_id).values
        numeric_arr[ids] = time_df['sec'].values.reshape(-1, 1)

    np.save(f"processed_data/{prefix}_{time_opt}_numeric_{num_patients}.npy", numeric_arr)
    print("Literals saved.")


def preprocess_kg(num_patients, input_dir: Path, output_dir: Path, prefix: str = "sphn_pc", time_opt="TS"):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = input_dir / f"{prefix}_{time_opt}_{num_patients}.nt"
    node_df = pd.read_csv(input_file, sep=" ", header=None)
    node_df.drop(columns=node_df.columns[-1], axis=1, inplace=True)
    node_df.columns=['h', 'r', 't']

    # Map id to entities and relations.
    ent_to_id = {k: v for v, k in enumerate(set(node_df['h']).union(set(node_df['t'])), start=0)}
    rel_to_id = {k: v for v, k in enumerate(set(node_df['r']), start=0)}

    triples = node_df.copy()
    triples["h"] = node_df.h.map(ent_to_id)
    triples["t"] = node_df.t.map(ent_to_id)
    triples["r"] = node_df.r.map(rel_to_id)    

    entity = pd.DataFrame({'id': list(ent_to_id.values()), 'entity': list(ent_to_id)})
    relation = pd.DataFrame({'id': list(rel_to_id.values()), 'relation': list(rel_to_id)})

    # Save triples, entities and relations.
    triples.to_csv(output_dir / f"{prefix}_{time_opt}_triples_{num_patients}.tsv", sep='\t', index=False, header=False)
    entity.to_csv(output_dir / f"{prefix}_{time_opt}_entities_{num_patients}.tsv", sep='\t', index=False, header=False)
    relation.to_csv(output_dir / f"{prefix}_{time_opt}_relations_{num_patients}.tsv", sep='\t', index=False, header=False)
    
    print("Triples / Entities / Relations saved successfully.")

    print(f"[Triples]: {len(triples)} - [Entity]: {len(entity)} - [Relation]: {len(relation)}")

    if prefix=="sphn_pc":
        preprocess_sphn_kg(node_df=node_df, entity=entity, time_opt=time_opt, num_patients=num_patients)
    elif prefix=="meds":
        preprocess_meds_kg(node_df=node_df, entity=entity, time_opt=time_opt, num_patients=num_patients)
