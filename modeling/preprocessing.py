#%%writefile /content/src/preprocessing.py
import numpy as np
import os
import pandas as pd
from glob import glob


def get_patients_from_paths(image_dir):
    patients = set()
    for x in glob(image_dir + '*ultra*.npy'):
        x = x.split('/')[-1]
        p = "-".join(x.split('-')[:-2])
        patients.add(p)

    return sorted(patients)


def init_patient_datafrmae(image_dir):
    patients = get_patients_from_paths(image_dir)

    df_patient = []
    for p in patients:
        for i in range(840):
            x_path = image_dir + f'{p}-ultra-{i}.npy'
            assert os.path.isfile(image_dir + f'{p}-ultra-{i}.npy')
            y_path = image_dir + f'{p}-mask-{i}.npy'
            label = ""
            if os.path.isfile(y_path):
                label = y_path

            row = {'patient': p, 'label': label, 'image': x_path, 'idx': i}
            df_patient.append(row)

    return pd.DataFrame(df_patient)


def load_mask_area(path):
    if not path:
        return 0

    return np.minimum(np.load(path), 1).sum()


def join_volume_labels(df_patient):
    df_patient['volume-label'] = None

    df_patient_ = []
    for p, pat_df in df_patient.groupby("patient"):

        y_areas = [load_mask_area(y) for y in pat_df.label.values]
        max_area = max(y_areas)
        y = [0 for _ in y_areas]
        if max_area > 0:
            y = []
            for a in y_areas:
                v = 0
                if a == max_area:
                    v = 1
                elif a > 0:
                    v = .6
                y.append(v)

        pat_df['volume-label'] = y
        df_patient_.append(pat_df)

    return pd.concat(df_patient_)


def create_patient_datafrmae(image_dir):
    # flow image paths from image_dir into a datafrmae
    df = init_patient_datafrmae(image_dir)

    # join labels
    df = join_volume_labels(df)

    return df
