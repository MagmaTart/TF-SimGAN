import cv2
import numpy as np

import os

class Loader:
    def __init__(self):
        self.dataset_path = './dataset/crop-images/'
        self.filenames = []
        self.num_examples = 0
        self.current_batch = 0

    def load_data(self):
        for _, _, files in os.walk(self.dataset_path):
            for filename in files:
                self.filenames.append(filename)

        self.num_examples = len(self.filenames)
        print(self.filenames[0])

    def get_next_batch(self, batch_size=128):
        self.current_batch = 0 if self.current_batch + batch_size >= self.num_examples else self.current_batch

        batch = [cv2.imread(self.dataset_path+self.filenames[i])
                 for i in range(self.current_batch, self.current_batch + batch_size)]
        batch = np.array(batch) / 127.5

        self.current_batch += batch_size

        return batch