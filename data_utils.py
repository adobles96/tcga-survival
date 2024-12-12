import os

import numpy as np
import pandas as pd
import requests
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import torch
from tqdm import tqdm

CASES_ENDPT = "https://api.gdc.cancer.gov/cases"
FILES_ENDPT = "https://api.gdc.cancer.gov/files"
SEED = 42


def pull_cases(project_ids: list[str] = ["TCGA-LUAD"]):

    fields = [
        "case_id",
        "disease_type",
        "primary_site",
        # demographics
        "demographic.gender",
        "demographic.race",
        "demographic.year_of_birth",
        "demographic.year_of_death",
        "demographic.days_to_death",
        "demographic.vital_status",
        # diagnoses
        "diagnoses.diagnosis_id",
        "diagnoses.updated_datetime",
        "diagnoses.age_at_diagnosis",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.days_to_recurrence",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.last_known_disease_status",
        "diagnoses.days_to_last_known_disease_status",
        "diagnoses.treatments.treatment_id",
        "diagnoses.treatments.days_to_treatment_start",
        "diagnoses.treatments.days_to_treatment_end",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.days_to_treatment",
        "diagnoses.treatments.therapeutic_agents",
    ]

    # Construct filters for WSI and molecular data
    filters = {
        "op": "in",
        "content": {
            "field": "cases.project.project_id",
            "value": project_ids
        }
    }

    params = {
        "filters": filters,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000"  # Adjust based on expected number of cases
    }

    # Make the API request
    response = requests.post(CASES_ENDPT, json=params)

    if response.status_code == 200:
        return response.json()['data']['hits']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")


def process_cases(cases):
    data = []
    for case in tqdm(cases, unit="cases", colour="green"):
        if 'demographic' not in case:
            print(f"Skipping case {case['case_id']} because it has no demographic data")
            continue
        if 'diagnoses' not in case:
            print(f"Skipping case {case['case_id']} because it has no diagnosis data")
            continue
        record = {
            'case_id': case['case_id'],
            'disease_type': case['disease_type'],
            'primary_site': case['primary_site'],
            # demographics
            'gender': case.get('demographic').get('gender'),
            'race': case.get('demographic').get('race'),
            'year_of_birth': case.get('demographic').get('year_of_birth'),
            'year_of_death': case.get('demographic').get('year_of_death'),
            'vital_status': case.get('demographic').get('vital_status'),
            'days_to_death': case.get('demographic').get('days_to_death'),
            # diagnoses
            'num_diagnoses': len(case.get('diagnoses')),
            'age_at_diagnosis': case.get('diagnoses')[0].get('age_at_diagnosis'),
            'days_to_diagnosis': case.get('demographic', {}).get('days_to_diagnosis'),
            'ajcc_pathologic_stage': case.get('diagnoses')[0].get('ajcc_pathologic_stage'),
            'days_to_last_follow_up': case.get('diagnoses')[0].get('days_to_last_follow_up'),
            'days_to_recurrence': case.get('diagnoses')[0].get('days_to_recurrence'),
            'days_to_last_known_disease_status': case.get('diagnoses')[0]\
            .get('days_to_last_known_disease_status'),
        }
        if str(record['vital_status']).lower() == 'dead':
            record['survival_time'] = record['days_to_death']
            record['event'] = 1
        else:
            record['survival_time'] = np.max(
                record['days_to_last_follow_up'],
                record['days_to_last_known_disease_status']
            )
            record['event'] = 0

        data.append(record)

    df = pd.DataFrame(data)

    # extra features for convenience
    df['ajcc_pathologic_stage_coarse'] = df['ajcc_pathologic_stage'].str.strip('AB')
    df['age_at_diagnosis_years'] = df['age_at_diagnosis'] // 365

    return df


def get_case_files(case_id: str | list[str]):
    if isinstance(case_id, str):
        case_id = [case_id]
    # Define the filters for the query
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.case_id",
                    "value": case_id
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": [
                        "Slide Image",
                        "Gene Expression Quantification",
                        # "Simple Germline Variation", NOTE this data format is controlled
                        "Gene Level Copy Number",
                    ]
                }
            }
        ]
    }

    # Define the parameters for the request
    params = {
        "filters": filters,
        "fields": (
            "cases.case_id,file_id,file_name,experimental_strategy,data_type,data_format,file_size"
        ),
        "format": "JSON",
        "size": "1000"
    }

    # Make the request to the GDC API
    response = requests.post(FILES_ENDPT, json=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        return data["data"]["hits"]
    else:
        response.raise_for_status()


def download_file(file_id, file_type, dir, file_size=None) -> str:
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    path = os.path.join(os.getcwd(), dir, file_id + f'.{file_type.lower()}')
    if os.path.exists(path):
        print('File already downloaded!')
        return path

    # Make the request to the GDC API
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        chunksize = 1024 * 1024  # 1 MB chunks
        total = (file_size // chunksize) + 1 if file_size is not None else None
        # Write the file content to the specified path
        with open(path, 'wb') as file:
            for chunk in tqdm(
                response.iter_content(chunk_size=chunksize), desc='Downloading file', total=total,
                unit='MB', leave=False
            ):
                file.write(chunk)
    else:
        response.raise_for_status()
    return path


def preprocess_demo(X_demo):
    X_demo = X_demo.copy(deep=True)
    cat_cols = ['disease_type', 'gender', 'race', 'ajcc_pathologic_stage_coarse']
    X_demo.loc[:, cat_cols] = X_demo[cat_cols].astype('category')
    X_demo = pd.get_dummies(X_demo, drop_first=True)
    X_demo.loc[X_demo.age_at_diagnosis_years.isna(), 'age_at_diagnosis_years'] = -1
    return X_demo.astype(float)


def preprocess_wsi(X_wsi, y, n_feats=10, n_components=50):
    assert n_feats <= n_components
    index = X_wsi.index
    # PCA
    X_wsi = pd.DataFrame(
        PCA(n_components=n_components, random_state=SEED).fit_transform(X_wsi),
        columns=[f'wsi_pca_{i}' for i in range(1, n_components + 1)],
        index=index
    )
    # Feature selection
    model_select = SelectFromModel(
        GradientBoostingSurvivalAnalysis(n_estimators=200, random_state=SEED), threshold=-np.inf,
        max_features=n_feats
    ).fit(X_wsi, y)
    return pd.DataFrame(
        model_select.transform(X_wsi),
        columns=model_select.get_feature_names_out(),
        index=index
    )


def preprocess_omics(X_omics, y, n_feats=10, n_components=50):
    assert n_feats <= n_components
    index = X_omics.index
    # PCA
    X_omics = pd.DataFrame(
        PCA(n_components=n_components, random_state=SEED).fit_transform(X_omics),
        columns=[f'omics_pca_{i}' for i in range(1, n_components + 1)],
        index=index
    )
    # Feature selection
    model_select = SelectFromModel(
        GradientBoostingSurvivalAnalysis(n_estimators=200, random_state=SEED), threshold=-np.inf,
        max_features=n_feats
    ).fit(X_omics, y)
    return pd.DataFrame(
        model_select.transform(X_omics),
        columns=model_select.get_feature_names_out(),
        index=index
    )


def load_dataset(
    case_file='data/case_data_filtered.csv', include_demo=False, include_wsi=False,
    include_omics=False, wsi_n_feats=10, omics_n_feats=10
):
    cases = pd.read_csv(case_file)
    cases = cases[cases['survival_time'] > 0].set_index('case_id')
    y = np.array(
        [((row['event'] == 1), row['survival_time']) for _, row in cases.iterrows()],
        dtype=[('event', bool), ('survival_time', float)]
    )
    if include_demo:
        demo_cols = ['disease_type', 'gender', 'race', 'age_at_diagnosis_years',
                     'ajcc_pathologic_stage_coarse']
        X_demo = cases[demo_cols]
        X_demo = preprocess_demo(X_demo)
    else:
        X_demo = None
    if include_wsi:
        embs = torch.load('data/wsi_embs1024.pkl')
        X_wsi = pd.DataFrame(
            {case_id: embs[case_id] for case_id in cases.index if case_id in embs}
        ).T
        X_wsi = preprocess_wsi(X_wsi, y, n_feats=wsi_n_feats, n_components=50)
    else:
        X_wsi = None
    if include_omics:
        embs = torch.load('data/gene_embs.pkl')
        X_omics = pd.DataFrame(
            {case_id: embs[case_id].numpy() for case_id in cases.index if case_id in embs}
        ).T
        X_omics = preprocess_omics(X_omics, y, n_feats=omics_n_feats, n_components=50)
    else:
        X_omics = None
    X = pd.concat([x for x in [X_demo, X_wsi, X_omics] if x is not None], join='inner', axis=1)
    return X, y


def download_genomics(case_file):
    cases = pd.read_csv(case_file)
    for case in tqdm(cases['case_id'], unit='cases', colour='green'):
        if os.path.exists(f'data/omics/gene_exp/{case}.tsv'):
            continue
        files = get_case_files(case)
        for file in files:
            if file['data_type'] == 'Gene Expression Quantification':
                path = download_file(file['file_id'], file['data_format'], 'data/omics/gene_exp',
                                     file['file_size'])
                # rename file to case_id
                os.rename(path, os.path.join(os.path.dirname(path), case + '.tsv'))
                break
        else:
            print(f'No gene expression data found for {case}.')


def compute_average_tpm_per_gene():
    files = os.listdir('data/omics/gene_exp')
    totals = None
    count = 0
    for file in tqdm(files, unit='files', colour='green'):
        df = pd.read_csv(
            os.path.join('data/omics/gene_exp', file), sep='\t', header=1, index_col='gene_name'
        )
        if totals is None:
            totals = df.tpm_unstranded
        else:
            totals += df.tpm_unstranded
        count += 1
    totals /= count
    totals.to_csv('data/average_tpm.csv', index=True)


def get_gene_embeds(top_k=10):
    gene_embs = torch.load('data/human_embedding.torch')
    avg = pd.read_csv('data/average_tpm.csv', index_col='gene_name')['tpm_unstranded']
    case_to_embed = {}
    for file in tqdm(os.listdir('data/omics/gene_exp'), unit='cases', colour='green'):
        case_id = file.split('.')[0]
        tpms = pd.read_csv(
            os.path.join('data/omics/gene_exp', file), sep='\t', header=1, index_col='gene_name'
        )['tpm_unstranded']
        tpms -= avg
        top_genes = tpms.sort_values(ascending=False).index[:top_k]
        embed = sum([gene_embs.get(gene, torch.zeros(5120)) for gene in top_genes]) / top_k
        case_to_embed[case_id] = embed
    torch.save(case_to_embed, 'data/gene_embs.pkl')


if __name__ == '__main__':
    download_genomics('data/case_data_filtered.csv')
    # compute_average_tpm_per_gene()
    # get_gene_embeds()
