import os
import zipfile
import json
import pandas as pd
from PIL import Image
import io
import requests
import torch
from datasets import Dataset, DatasetDict, Features, Value

class POPP_Dataset:
    def __init__(self, zip_path=None, fraction=1.0, download_url="https://zenodo.org/records/6581158/files/popp.zip?download=1"):
        self.zip_path = zip_path
        self.download_url = download_url
        self.dataset_dict = None
        self.data = {}
        self.fraction = fraction
        self._prepare_data()
        self._load_labels()
        self._create_datasets()

    def _prepare_data(self):
        """
        If zip_path is not provided, download the dataset. Then extract it.
        """
        if self.zip_path is None:
            self.zip_path = 'popp.zip'
            if not os.path.exists(self.zip_path):
                print(f"Downloading dataset from {self.download_url}...")
                self._download_dataset()
        else:
            if not os.path.exists(self.zip_path):
                print(f"Dataset not found at {self.zip_path}. Downloading...")
                self._download_dataset()
        self._extract_data()

    def _download_dataset(self):
        """
        Download the dataset from the specified URL without a progress bar.
        """
        response = requests.get(self.download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KiB
        downloaded_size = 0

        with open(self.zip_path, 'wb') as file:
            for data in response.iter_content(block_size):
                size = file.write(data)
                downloaded_size += size

        if total_size != 0 and downloaded_size != total_size:
            print("ERROR: Something went wrong during the download.")
            raise Exception("Download failed.")
        else:
            print("Download complete.")

    def _extract_data(self):
        """
        Extract the contents of the ZIP file to the current directory.
        """
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction complete.")

    def _load_labels(self):
        """
        Load the labels from each labels.json file.
        """
        for location in ['Belleville', 'ChausseeDAntin', 'Generic']:
            labels_path = os.path.join('data', location, 'lines', 'labels.json')
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
                self.data[location] = labels['ground_truth']

    def _load_image(self, image_path):
      """
      Load an image from a given path, convert it to RGB, and return it as bytes.
      """
      try:
          image = Image.open(image_path)
          image.load()  # Ensure the image is fully loaded
          image = image.convert('RGB')
          img_byte_arr = io.BytesIO()
          image.save(img_byte_arr, format='PNG')
          img_byte_arr = img_byte_arr.getvalue()
          return img_byte_arr
      except Exception as e:
          print(f"Error loading image {image_path}: {e}")
          return None

    def _create_datasets(self):
      """
      Create HuggingFace Datasets for train, valid, and test splits with image bytes.
      """
      dataset_splits = {'train': [], 'valid': [], 'test': []}

      for location, data in self.data.items():
          for split in ['train', 'valid', 'test']:
              split_data = data.get(split, {})
              split_data_items = list(split_data.items())
              num_samples = max(1, int(len(split_data_items) * self.fraction))
              for image_name, info in split_data_items[:num_samples]:
                  image_path = os.path.join('data', location, 'lines', split, image_name)
                  if os.path.exists(image_path):
                      image_bytes = self._load_image(image_path)
                      if image_bytes is not None:
                          dataset_splits[split].append({
                              'image_name': image_name,
                              'image': image_bytes,  # Store image bytes
                              'text': info['text']
                          })
                      else:
                          print(f"Skipped image {image_path} due to loading error.")
                  else:
                      print(f"Image not found: {image_path}")

      for split, data in dataset_splits.items():
          print(f"Number of items in {split} split: {len(data)}")

      # Define the features, specifying that 'image' is a binary feature
      features = Features({
          'image_name': Value('string'),
          'image': Value('binary'),  # Specify binary feature for image bytes
          'text': Value('string'),
      })

      # Create the DatasetDict with the specified features
      self.dataset_dict = DatasetDict({
          split: Dataset.from_pandas(pd.DataFrame(data), features=features)
          for split, data in dataset_splits.items()
      })

    def get_datasets(self):
        """
        Get the HuggingFace DatasetDict containing train, valid, and test splits.
        """import os
import zipfile
import json
import pandas as pd
from PIL import Image
import io
import requests
import torch
from datasets import Dataset, DatasetDict, Features, Value
        return self.dataset_dict