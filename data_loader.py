import h5py
import random
import numpy as np

class DataLoader:
    def __init__(self, h5_path, dso_dict, batch_size):
        self.h5_path = h5_path
        self.dso_dict = dso_dict
        self.batch_size = batch_size

        self.groups = {} # ==> {'dso class1': [record1, record2, ...], 'dso class2': [...], ...}
        with h5py.File(self.h5_path, 'r') as f:
            for group_name in f.keys():
                print(f"Found DSO group in hdf5: {group_name}")
                if group_name in self.dso_dict:
                    self.groups[group_name] = list(f[group_name].keys()) 
        
        print('==\n*dso: the number of keys in each class')
        total_keys = 0
        for cls, keys in self.groups.items():
            num_keys = len(keys)
            total_keys += num_keys
            print(f"{cls}: {num_keys} keys")
        print(f"Total keys: {total_keys}\n==")
                    
        self.dso_class_names = list(self.groups.keys())
        num_classes = len(self.dso_class_names) # ==> 3
        if batch_size % num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")
        self.per_class = batch_size // num_classes # ==> 300 // 3
        
        print(f"Data loader initialized with {num_classes} dso classes and {self.per_class} samples per class.")

    def get_batch(self):
        acc2d_batch = []
        tx2d_batch = []
        dso_batch = []
        
        with h5py.File(self.h5_path, 'r') as f:
            # For each dso class (3 classes), randomly select self.per_class (e.g., 100) records 
            for cls in self.dso_class_names:
                keys = self.groups[cls]
                selected_keys = random.sample(keys, k=self.per_class)
                
                for key in selected_keys:
                    rec = f[cls][key]
                    
                    acc2d = rec['acc2d'][()] # (64, 64)
                    tx2d = rec['tx2d'][()] # (64, 64)
                    dso = self.dso_dict[cls]['dso'] # (200,)
                    
                    acc2d = np.reshape(acc2d, (64, 64, 1))
                    tx2d = np.reshape(tx2d, (64, 64, 1))
                    
                    acc2d_batch.append(acc2d.astype(np.float32))
                    tx2d_batch.append(tx2d.astype(np.float32))
                    dso_batch.append(dso.astype(np.float32))
        
        # Stack ==> shuffle 
        acc2d_batch = np.stack(acc2d_batch, axis=0)
        tx2d_batch = np.stack(tx2d_batch, axis=0)
        dso_batch = np.stack(dso_batch, axis=0)
        
        num_samples = acc2d_batch.shape[0]
        perm = np.random.permutation(num_samples)

        acc2d_batch = acc2d_batch[perm] # ==> (batch_size (e.g., 300), 64, 64, 1)
        tx2d_batch = tx2d_batch[perm] # ==> (batch_size, 64, 64, 1)
        dso_batch = dso_batch[perm] # ==> (batch_size, 200)
        
        return acc2d_batch, tx2d_batch, dso_batch