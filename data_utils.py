import os

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

CASES_ENDPT = "https://api.gdc.cancer.gov/cases"
FILES_ENDPT = "https://api.gdc.cancer.gov/files"


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


def download_file(file_id, file_type, dir, file_size=None):
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    path = os.path.join(os.getcwd(), dir, file_id + f'.{file_type.lower()}')
    chunksize = 16_384

    # Make the request to the GDC API
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        total = (file_size // chunksize) + 1 if file_size is not None else None
        # Write the file content to the specified path
        with open(path, 'wb') as file:
            for chunk in tqdm(
                response.iter_content(chunk_size=chunksize), desc='Downloading file', total=total
            ):
                file.write(chunk)
    else:
        response.raise_for_status()
