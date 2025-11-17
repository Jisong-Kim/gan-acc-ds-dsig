import os
import glob
import numpy as np
from easydict import EasyDict as edict


def build_paths(data, dso, savepth):
    """
    Paths configuration
    Required data and folder structures:

    1. data: path to the HDF5 data file,
    e.g., data.h5:
        ├─ [Group] ASCE41-17
        │   ├─ [Group] record1
        │   │   ├─ [Dataset] acc2d: shape: (64, 64)
        │   │   └─ [Dataset] tx2d: shape: (64, 64)
        │   ├─ [Group] record2
        │   │   ├─ [Dataset] acc2d
        │   │   └─ [Dataset] tx2d
        │   └─ ...
        │
        ├─ [Group] KDS 17 10 00: 2018
        │   ├─ [Group] record1
        │   │   ├─ [Dataset] acc2d
        │   │   └─ [Dataset] tx2d
        │   ├─ [Group] record2
        │   │   ├─ [Dataset] acc2d
        │   │   └─ [Dataset] tx2d
        │   └─ ...
        │
        └─ [Group] Notification 1457-2000
            ├─ [Group] record1
            │   ├─ [Dataset] acc2d
            │   └─ [Dataset] tx2d
            ├─ [Group] record2
            │   ├─ [Dataset] acc2d
            │   └─ [Dataset] tx2d
            └─ ...

    2. dso: directory path that contains the design spectrum csv files,
    e.g., dso_directory/
        ├─ ASCE41-17.csv
        ├─ KDS 17 10 00 2018.csv
        └─ Notification 1457-2000.csv
        Each file contains two columns: Period (s), Sa (g)
    
    3. savepth: path to save the trained models    
    """
    
    return edict({
        "data": data,
        "dso": dso,
        "savepth": savepth})
    

def load_dso(path_dso):
    # Read the design spectra data from the csv files
    targets = glob.glob(os.path.join(path_dso, '*.csv'))
    targets.sort()
    targets_basename = [os.path.basename(file).replace('.csv', '') for file in targets]

    target_dict = {}
    for target, basename in zip(targets, targets_basename):
        tso = np.loadtxt(target, delimiter=',', skiprows=1)
        To = tso[:, 0]  # periods (0.02, 0.04, ..., 4.0s)
        dso = tso[:, 1]  # SA
        target_dict[basename] = {
            'target': target,
            'To': To,
            'dso': dso
        }
        
    return target_dict

def get_config():
    cfg = edict()
    cfg.dso_dim = 200
    cfg.batch_size = 3 * 100 # Three design spectrum, 100 samples each
    cfg.total_iters = 50000
    cfg.n_critic = 5
    cfg.lambda_gp = 10.0
    
    cfg.lr = 1e-4
    cfg.beta_1 = 0.5
    cfg.beta_2 = 0.9
    
    return cfg