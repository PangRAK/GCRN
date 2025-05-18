import optparse
import multiprocessing as mp

import tqdm

import numpy as np

import os
import h5py
from moviepy.editor import *
import tqdm

import torch.utils.data as td
import torch

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

import pdb

AVE_DATASET = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', \
                    'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar', \
                    'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw',\
                    'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin', 'Background']

class AVEDataset(td.Dataset):

    def __init__(self,
                 root: str,
                 sample_rate: int = 22050,
                 train: bool = True,
                 split: str = 'train',
                 fold: Optional[int] = None,
                 transform_audio=None,
                 transform_frames=None,
                 target_transform=None,
                 **_):

        super(AVEDataset, self).__init__()
        
        self.sample_rate = sample_rate
        fold = 1
        self.folds_to_load = set([1])
        if fold not in self.folds_to_load:
            raise ValueError(f'fold {fold} does not exist')

        self.split = split
    
        # set data path
        self.visual_feature_path = os.path.join(root, 'CFT_visual_features_float32.h5')
        self.audio_feature_path = os.path.join(root, 'CFT_audio_features_float32.h5')
        # self.visual_feature_path = os.path.join(root, 'audioclip_visual_features_float32.h5')
        # self.audio_feature_path = os.path.join(root, 'audioclip_audio_features_float32.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(root, 'raw_labels_data.h5')
        self.weak_labels_path = os.path.join(root, 'raw_weak_labels_data.h5')
        self.file_names_path = os.path.join(root, 'raw_file_names.h5')
        self.sample_order_path = os.path.join(root, f'data_order.h5')
        # self.sample_order_path = os.path.join(root, f'{self.split}_order.h5')

        # load data
        self.load_data()

    def load_data(self):
        self.visual_feature = h5py.File(self.visual_feature_path, 'r')['data']
        self.audio_feature = h5py.File(self.audio_feature_path, 'r')['data']
        self.labels = h5py.File(self.labels_path, 'r')['data']
        self.weak_labels = h5py.File(self.weak_labels_path, 'r')['data']
        self.file_names = h5py.File(self.file_names_path, 'r')['data']
        self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not (0 <= index < len(self)):
            raise IndexError
        sample_index = self.sample_order[index]
        labels: str = self.labels[sample_index]    
        audio: np.ndarray = self.audio_feature[sample_index]    # (10, 1024)
        image: np.ndarray = self.visual_feature[sample_index]   # (10, 1024)
        target: str = self.weak_labels[sample_index]
        file_name: str = self.file_names[sample_index]
        
        target = target.decode('utf-8')

        audio = audio.astype(np.float32)
        image = image.astype(np.float32)
        # audio = torch.tensor(audio)
        # image = torch.tensor(image)

        return image, audio, labels, file_name
        # return image, audio, [target]

    def __len__(self) -> int:
        return len(self.sample_order)
    
    def __del__(self):
        pass


# import optparse
# import multiprocessing as mp

# import tqdm

# import numpy as np

# import os
# import h5py
# from moviepy.editor import *
# import tqdm

# import torch.utils.data as td
# import torch

# from typing import Any
# from typing import Dict
# from typing import List
# from typing import Tuple
# from typing import Union
# from typing import Optional

# import pdb

# AVE_DATASET = ['Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane', 'Race car, auto racing', \
#                     'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute', 'Ukulele', 'Frying (food)', 'Truck', 'Shofar', \
#                     'Motorcycle', 'Acoustic guitar', 'Train horn', 'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw',\
#                     'Cat', 'Horse', 'Toilet flush', 'Rodents, rats, mice', 'Accordion', 'Mandolin', 'Background']

# class AVEDataset(td.Dataset):

#     def __init__(self,
#                  root: str,
#                  sample_rate: int = 22050,
#                  train: bool = True,
#                  split: str = 'train',
#                  fold: Optional[int] = None,
#                  transform_audio=None,
#                  transform_frames=None,
#                  target_transform=None,
#                  **_):

#         super(AVEDataset, self).__init__()
        
#         self.sample_rate = sample_rate
#         fold = 1
#         self.folds_to_load = set([1])
#         if fold not in self.folds_to_load:
#             raise ValueError(f'fold {fold} does not exist')

#         self.split = split
    
#         # set data path
#         self.visual_feature_path = os.path.join(root, 'CFT_visual_features.h5')
#         self.audio_feature_path = os.path.join(root, 'CFT_audio_features.h5')
#         # Now for the supervised task
#         self.labels_path = os.path.join(root, 'raw_labels_data.h5')
#         self.weak_labels_path = os.path.join(root, 'raw_weak_labels_data.h5')
#         self.sample_order_path = os.path.join(root, f'data_order.h5')
#         # self.sample_order_path = os.path.join(root, f'{self.split}_order.h5')

#         # load data
#         self.load_data()

#     def load_data(self):
#         self.visual_feature = h5py.File(self.visual_feature_path, 'r')['data']
#         self.audio_feature = h5py.File(self.audio_feature_path, 'r')['data']
#         self.labels = h5py.File(self.labels_path, 'r')['data']
#         self.weak_labels = h5py.File(self.weak_labels_path, 'r')['data']
#         self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
        
#     def __getitem__(self, index: int) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
#         if not (0 <= index < len(self)):
#             raise IndexError

#         sample_index = self.sample_order[index]
#         labels: str = self.labels[sample_index]    
#         audio: np.ndarray = self.audio_feature[sample_index]    # (10, 1024)
#         image: np.ndarray = self.visual_feature[sample_index]   # (10, 1024)
#         target: str = self.weak_labels[sample_index]
        
#         target = target.decode('utf-8')

#         audio = audio.astype(np.float32)
#         image = image.astype(np.float32)
#         # audio = torch.tensor(audio)
#         # image = torch.tensor(image)

#         return image, audio, labels
#         # return image, audio, [target]

#     def __len__(self) -> int:
#         return len(self.sample_order)
    
#     def __del__(self):
#         pass
