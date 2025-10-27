import torch
from torch.utils.data import Dataset    
import os
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive, download_url
import json
from transformers import BertTokenizer
import urllib.request
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split

# CLASSES = {"mnist": 10, "cifar": 10, "cifar100": 100, "emnist": 47, "fashionmnist": 10}
CLASSES = {
    "mnist": 10,
    "cifar": 10,
    "cifar100": 100,
    "emnist": 47,
    "fmnist": 10,
    "femnist": 62,
    "har": 6,              # Common HAR datasets (e.g., UCI HAR) have 6 activity classes
    "areview": 2,     # Usually binary sentiment classification (positive/negative)
}



class MNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class HAR(Dataset):
    har_link = "https://archive.ics.uci.edu/static/public/244/human+activity+recognition+using+smartphones.zip"

    def __init__(self, root, transform=None, target_transform=None, download=False, train=True):
        self.root = os.path.join(root, "UCI HAR Dataset")
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        self.data = []
        self.targets = []
        self._load_data()

    def download(self):
        if os.path.exists(self.root):
            print("HAR dataset already downloaded.")
            return
        os.makedirs(self.root, exist_ok=True)
        print("Downloading HAR dataset...")
        download_and_extract_archive(self.har_link, download_root=os.path.dirname(self.root), extract_root=os.path.dirname(self.root), remove_finished=True)

    def _load_data(self):
        split = "train" if self.train else "test"
        x_path = os.path.join(self.root, split, f"X_{split}.txt")
        y_path = os.path.join(self.root, split, f"y_{split}.txt")

        assert os.path.exists(x_path), f"{x_path} not found"
        assert os.path.exists(y_path), f"{y_path} not found"

        # Load as float32 tensor
        x_data = np.loadtxt(x_path).astype(np.float32)
        y_data = np.loadtxt(y_path).astype(int) - 1  # classes are 1-indexed

        self.data = torch.tensor(x_data).reshape(-1, 561)  # Shape: [N, 561]
        self.targets = torch.tensor(y_data).long()

    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.targets)

    

class HARDataset(Dataset):
    def __init__(
        self,
        data=None,
        targets=None,
        subset=None,
        transform=None,
        target_transform=None,
    ):
        self.transform = transform
        self.target_transform = target_transform

        if data is not None and targets is not None:
            self.data = data  # Shape: (N, T, F)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                [torch.tensor(tup[0]) if not isinstance(tup[0], torch.Tensor) else tup[0] for tup in subset]
            )
            self.targets = torch.tensor([tup[1] for tup in subset])
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / np.ndarray, targets: Tensor) OR data: Tensor  targets: Tensor"
            )

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.targets)
    

'''class AmazonReview(Dataset):
    amazon_link = "https://raw.githubusercontent.com/gsarti/amazon-review-spam/main/data/amazon_reviews_2class_small.json"

    def __init__(self, root, split="train", tokenizer=None, max_length=128, download=False, test_size=0.2, seed=42):
        self.root = root
        self.split = split
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.test_size = test_size
        self.seed = seed

        self.full_data_path = os.path.join(self.root, "amazon_reviews_2class_small.json")
        self.split_data_path = os.path.join(self.root, f"{split}.json")

        if download:
            self.download_and_split()

        self._load_data()

    def download_and_split(self):
        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(self.full_data_path):
            print("Downloading full Amazon Review dataset...")
            download_url(self.amazon_link, self.root, filename="amazon_reviews_2class_small.json")
        else:
            print("Full Amazon Review dataset already exists.")

        if not os.path.exists(self.split_data_path):
            print(f"Creating {self.split}.json split...")
            with open(self.full_data_path, "r") as f:
                all_data = json.load(f)

            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                all_data, test_size=self.test_size, random_state=self.seed, stratify=[x["label"] for x in all_data]
            )

            split_data = train_data if self.split == "train" else test_data
            with open(self.split_data_path, "w") as f:
                json.dump(split_data, f)
        else:
            print(f"{self.split}.json already exists.")

    def _load_data(self):
        with open(self.split_data_path, "r") as f:
            raw_data = json.load(f)

        self.texts = [x["text"] for x in raw_data]
        self.labels = [x["label"] for x in raw_data]

        self.encodings = self.tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return item, label

    def __len__(self):
        return len(self.labels)

class AmazonReview(Dataset):
    """
    Uses the Stanford SNAP Amazon Reviews subset via OpenML.
    Provides automatic train/test split based on known sizes.
    """
    openml_id = 46698  # Amazon reviews binary subset

    def __init__(self, root, split="train", tokenizer=None, max_length=128, download=False):
        self.root = root
        self.split = split
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        self.train_path = os.path.join(self.root, "amazon_train.csv")
        self.test_path = os.path.join(self.root, "amazon_test.csv")

        if download:
            self.download_and_prepare()

        self._load_data()

    def download_and_prepare(self):
        os.makedirs(self.root, exist_ok=True)

        import openml

        print("Fetching Amazon reviews dataset from OpenML...")
        # List datasets
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        print(datasets.head())

        # Download a specific dataset by ID
        dataset = openml.datasets.get_dataset(45104)  # Amazon reviews (spam detection)

        
        # dataset = OpenML.get_dataset(self.openml_id)
        df, *_ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        df = df.rename(columns={"polarity": "label"})
        df["label"] = df["label"].map({1: 0, 2: 1})  # zero-indexed classes

        # Stratified split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=200_000 / len(df), random_state=42, stratify=df["label"]
        )

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

    def _load_data(self):
        import pandas as pd

        path = self.train_path if self.split == "train" else self.test_path
        df = pd.read_csv(path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        self.encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels) #'''


class AmazonReview_(Dataset):
    """
    Amazon Reviews Binary Sentiment Dataset from OpenML (ID 46698).
    Automatically downloads and splits into train/test with tokenizer support.
    """

    openml_id = 46698  # OpenML dataset ID for binary sentiment classification

    def __init__(self, root, split="train", tokenizer=None, max_length=128, download=False):
        self.root = root
        self.split = split
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.openml_id = 45104

        self.train_path = os.path.join(self.root, "amazon_train.csv")
        self.test_path = os.path.join(self.root, "amazon_test.csv")

        if download:
            self.download_and_prepare()

        self._load_data()

    def download_and_prepare_old(self):
        os.makedirs(self.root, exist_ok=True)

        import openml
        from sklearn.model_selection import train_test_split
        import pandas as pd

        print("Downloading Amazon Reviews dataset from OpenML...")
        dataset = openml.datasets.get_dataset(self.openml_id)
        
        df, *_ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute
        )

        # Show available columns and target column for debugging
        print(f"Columns: {df.columns.tolist()}")
        print(f"Target (label) column: {dataset.default_target_attribute}")

        # Rename target column to "label"
        df = df.rename(columns={dataset.default_target_attribute: "label"})

        # Normalize label to 0/1 if necessary
        if df["label"].dtype == object:
            df["label"] = df["label"].map({"negative": 0, "positive": 1})
        else:
            df["label"] = df["label"].map({1: 0, 2: 1})  # only if applicable

        # Determine text column
        text_col = None
        for col in ["review", "content", "text"]:
            if col in df.columns:
                text_col = col
                break

        if not text_col:
            raise ValueError("No valid text column found in dataset.")

        df = df.rename(columns={text_col: "text"})

        # Split and save
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        print("Saved train/test splits.")

    def download_and_prepare_2(self):
        os.makedirs(self.root, exist_ok=True)

        import openml
        from sklearn.model_selection import train_test_split
        print("Downloading Amazon Reviews dataset from OpenML...")
        dataset = openml.datasets.get_dataset(45104)  # or your chosen dataset id

        df, *_ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
        
        # Rename target column 'polarity' to 'label' to match rest of code
        df = df.rename(columns={"polarity": "label"})

        # Check if 'label' column exists now
        print("Columns after renaming:", df.columns)

        # Map labels from {1, 2} to {0, 1}
        df["label"] = df["label"].map({1: 0, 2: 1})

        # Your train/test split and saving as csv follows
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        print("Saved train/test splits.")

    def download_and_prepare(self):
        os.makedirs(self.root, exist_ok=True)

        import openml
        print("Downloading Amazon Reviews dataset from OpenML...")

        dataset = openml.datasets.get_dataset(46698)
        df, *_ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        print("==== Dataset Preview ====")
        print(df.head())
        print("\n==== Columns ====")
        print(df.columns)
        print("\n==== Target column according to OpenML ====")
        print(dataset.default_target_attribute)

        print(f"Columns: {df.columns.tolist()}")
        print(f"Target (label) column: {dataset.default_target_attribute}")

        # Rename if needed
        if "polarity" in df.columns:
            df = df.rename(columns={"polarity": "label"})

        if "label" not in df.columns:
            raise KeyError("Expected a 'label' column but none found after renaming. Check dataset structure.")

        # Convert label to 0/1 (optional depending on data)
        if df["label"].dtype == object or df["label"].nunique() != 2:
            print(f"Unique label values before mapping: {df['label'].unique()}")
            label_map = {v: i for i, v in enumerate(sorted(df["label"].unique()))}
            df["label"] = df["label"].map(label_map)

        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=200_000 / len(df), random_state=42, stratify=df["label"]
        )

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)

    def _load_data(self):
        import pandas as pd

        path = self.train_path if self.split == "train" else self.test_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Missing required columns in {path}")

        texts = df["text"].tolist()
        labels = df["label"].tolist()

        self.encodings = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)


class AmazonReview__(Dataset):
    """
    Amazon Review Dataset using Kaggle source.
    Automatically downloads and prepares train/test splits.
    Supports optional HuggingFace-style tokenization.
    """
    def __init__(self, root="./data/areview", split="train", tokenizer=None, max_length=512, download=True):
        """
        Args:
            root: Directory to store the dataset.
            split: 'train' or 'test'.
            tokenizer: Optional tokenizer (e.g., from HuggingFace).
            max_length: Max sequence length for tokenized inputs.
            download: If True, will download the Kaggle dataset.
        """
        self.root = root
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_file = os.path.join(self.root, f"{split}.csv")

        if download:
            self.download_and_prepare()

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Expected file not found: {self.data_file}")

        self.df = pd.read_csv(self.data_file)

        if 'text' not in self.df.columns or 'polarity' not in self.df.columns:
            raise ValueError("CSV must contain 'text' and 'polarity' columns")

        self.df.rename(columns={"polarity": "label"}, inplace=True)

        if self.df['label'].dtype == object:
            self.label_map = {label: idx for idx, label in enumerate(self.df['label'].unique())}
            self.df['label'] = self.df['label'].map(self.label_map)
        else:
            self.label_map = None

    def download_and_prepare(self):
        os.makedirs(self.root, exist_ok=True)

        try:
            import kagglehub
        except ImportError:
            raise ImportError("Please install `kagglehub` via `pip install kagglehub`")

        print("Downloading Amazon Reviews dataset from Kaggle...")
        zip_path = kagglehub.dataset_download("kritanjalijain/amazon-reviews", path=self.root)
        print("Path to dataset files:", zip_path)
        # Unzip the dataset manually
        import zipfile
        import glob

        zip_files = glob.glob(os.path.join(zip_path, "*.zip"))
        if not zip_files:
            raise FileNotFoundError("No zip files found after download.")

        for zfile in zip_files:
            print(f"Extracting: {zfile}")
            with zipfile.ZipFile(zfile, 'r') as zip_ref:
                zip_ref.extractall(self.root)

        print("Dataset extracted to:", self.root)
        print("Expected files: train.csv, test.csv")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Remove batch dim
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            item['labels'] = torch.tensor(label)
            return item
        else:
            return text, label


class AmazonReview(Dataset):
    """
    Loads the Amazon Review Polarity dataset from FastAI S3.
    Automatically downloads, extracts, and preprocesses the data.
    """

    def __init__(self, root, split="train", tokenizer=None, max_length=128, download=False):
        self.root = root
        self.split = split  # "train" or "test"
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        self.train_path = os.path.join(self.root, "amazon_train.csv")
        self.test_path = os.path.join(self.root, "amazon_test.csv")

        if download:
            self.download_and_prepare()

        self._load_data()

    def download_and_prepare(self):
        os.makedirs(self.root, exist_ok=True)

        url = "https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz"
        tgz_path = os.path.join(self.root, "amazon_review_polarity_csv.tgz")
        extracted_dir = os.path.join(self.root, "amazon_review_polarity_csv")

        if not os.path.exists(tgz_path):
            print("Downloading Amazon Review dataset...")
            urllib.request.urlretrieve(url, tgz_path)
            print("Download completed.")

        if not os.path.exists(extracted_dir):
            print("Extracting dataset...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=self.root)
            print("Extraction completed.")

        train_csv = os.path.join(extracted_dir, "train.csv")
        test_csv = os.path.join(extracted_dir, "test.csv")

        # Add headers since CSV files have none
        col_names = ["label", "text1", "text2"]
        df_train = pd.read_csv(train_csv, header=None, names=col_names)
        df_test = pd.read_csv(test_csv, header=None, names=col_names)

        # Convert labels: 1 → 0 (negative), 2 → 1 (positive)
        df_train["label"] = df_train["label"].map({1: 0, 2: 1})
        df_test["label"] = df_test["label"].map({1: 0, 2: 1})

        # Save processed CSVs
        df_train.to_csv(self.train_path, index=False)
        df_test.to_csv(self.test_path, index=False)

    def _load_data(self):
        path = self.train_path if self.split == "train" else self.test_path
        df = pd.read_csv(path)

        self.texts = df["text2"].tolist()
        self.labels = df["label"].tolist()
        print("Tokenizing Texts.........")
        encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        self.encodings = encodings
        print(f"Labels at index 0: {self.labels[0]}")
        for l,i in enumerate(self.labels):
            if not isinstance(l, int) or abs(l) > 9223372036854775807:
                print(f"Invalid label: {l}, Index: {i}")

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class AmazonReviewDataset(Dataset):
    def __init__(
        self,
        encodings=None,  # Dict with keys: input_ids, attention_mask, (optional: token_type_ids)
        labels=None,     # Tensor/List of labels
        subset=None,
        transform=None,
        target_transform=None,
    ):
        self.transform = transform
        self.target_transform = target_transform

        if encodings and labels is not None:
            self.encodings = encodings
            self.labels = labels
        elif subset is not None:
            self.encodings = {
                key: torch.stack([torch.tensor(example[0][key]) for example in subset])
                for key in subset[0][0]
            }
            self.labels = torch.tensor([example[1] for example in subset])
        else:
            raise ValueError(
                "Expected either (encodings & labels) or subset=[({'input_ids':..., ...}, label)]"
            )

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        label = self.labels[idx]

        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            label = self.target_transform(label)

        return item, label

    def __len__(self):
        return len(self.labels)


class CIFARDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)


class CIFAR100Dataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)


class EMNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


# Optional alias for FEMNIST (EMNIST byclass, 62 classes) — structure identical to EMNISTDataset
class FEMNISTDataset(EMNISTDataset):
    pass
