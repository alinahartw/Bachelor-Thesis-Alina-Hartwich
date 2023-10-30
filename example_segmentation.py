from claspy.segmentation import BinaryClaSPSegmentation
from claspy.scoring import f1_score
from covering import covering
from covering import f_measure
import matplotlib.pyplot as plt

import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

from claspy.window_multi import suss_for_each_dim


def load_mosad_dataset():
    cp_filename = ABS_PATH + "\mosad_change_points.txt"
    cp_file = []

    with open(cp_filename, 'r') as file:
        for line in file.readlines(): cp_file.append(line.split(","))
        
    activity_filename = ABS_PATH + "\mosad_activities.txt"
    activities = dict()

    with open(activity_filename, 'r') as file:
        for line in file.readlines():
            line = line.split(",")
            routine, motions = line[0], line[1:]
            activities[routine] = [motion.replace("\n", "") for motion in motions]

    ts_filename = ABS_PATH + "\mosad_data.npz"
    T = np.load(file=ts_filename)

    df = []

    for row in cp_file:
        (ts_name, sample_rate), change_points = row[:2], row[2:]
        routine, subject, sensor = ts_name.split("_")
        ts = T[ts_name]

        df.append((ts_name, int(routine[-1]), int(subject[-1]), sensor, int(sample_rate), np.array([int(_) for _ in change_points]), np.array(activities[routine[-1]]), ts))

    return pd.DataFrame.from_records(df, columns=["dataset", "routine", "subject", "sensor", "sample_rate", "change_points", "activities", "time_series"])

def get_multi_ts(dataset, start):
    ts_len = len(dataset['time_series'].loc[dataset.index[start]] ) 
    multi_ts = np.zeros(shape = (9,ts_len), dtype = np.float64)

    for i in range(start, start+9):
        multi_ts[i-start] = dataset['time_series'].loc[dataset.index[i]]

    return multi_ts


mosad = load_mosad_dataset()

multivariate_ts = get_multi_ts(mosad, 54)
clasp = BinaryClaSPSegmentation(window_size = "min")
cps = clasp.fit_predict(multivariate_ts)
true = mosad['change_points'].loc[mosad.index[54]] 
clasp.plot(gt_cps=true, ts_name=[ "x-acc", "y-acc", "z-acc","x-gyro", "y-gyro", "z-gyro", "x-mag", "y-mag", "z-mag"], file_path="segmentation_example.png")