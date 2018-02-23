import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize


class AugmentedImageGenerator(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        dataset_df = pd.read_csv(dataset_csv_file)
        dataset_df = dataset_df.sample(frac=1.)
        self.x_path, self.y = dataset_df["Image Index"].as_matrix(), dataset_df[class_names].as_matrix()
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose

    def __bool__(self):
        return True

    def __len__(self):
        return int(np.ceil(len(self.x_path) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self._load_image(x_path) for x_path in batch_x_path])
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [np.array(y) for y in batch_y.T.tolist()]

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std

        if self.verbose > 0:
            print(f"generate batch_x (shape: {batch_x.shape})")
            print(f"generate batch_y (len: {len(batch_y)}, shape: {batch_y[0].shape})")

        if self.augmenter is not None:
            return self.augmenter.augment_images(batch_x), batch_y
        return batch_x, batch_y

    def _load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image = Image.open(image_path)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, self.target_size)
        return image_array
