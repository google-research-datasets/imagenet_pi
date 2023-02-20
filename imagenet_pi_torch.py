import os
import torch

from typing import Any, Dict, List, Optional, Tuple
from torchvision.datasets import ImageNet


def _to_float(x: List[str]) -> List[float]:
    return [float(i) for i in x]


class ImageNetPI(ImageNet):
    """ImageNet-PI dataset.

    ImageNet-PI is a relabelled version of the standard ILSVRC2012 ImageNet
    dataset in which the labels are provided by a collection of 16 deep neural
    networks with different architectures pre-trained on the standard
    ILSVRC2012. Specifically, the pre-trained models are downloaded from
    tf.keras.applications.

    On top of the new labels, ImageNet-PI also provides meta-data about the
    annotation process in the form of confidences of the models on their labels
    and additional information about each model.

    For more information see:
        https://github.com/google-research-datasets/imagenet_pi.


    This class wraps the torchvision.datasets.ImageNet class and modifies the
    __getitem__ method to return the labels and confidences of the annotators
    besides the standard image and clean target.

    Args:
        pi_root (str): Path to the root directory storing the annotator labels
            and confidences. The class expects a directory structure as follows:

                pi_root/
                    labels/
                        train.csv
                        validation.csv
                    confidences/
                        train.csv
                        validation.csv

        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or
            ``val``.
        transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): A function to load an image given its
            path.
    """

    def __init__(
        self,
        pi_root: str,
        root: str,
        split: str = "train",
        download: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, split, download, **kwargs)
        self.pi_root = pi_root
        self.annotator_labels = self._load_annotator_labels()
        self.annotator_confidences = self._load_annotator_confidences()
        print("Done initializing ImageNet-PI dataset.")

    def _load_annotator_labels(self) -> Dict[str, List[str]]:
        """Load the labels given by each annotator to every image into a dictionary.

        The labels are stored in a csv file with format:
            <image_path> <label_1> <label_2> ...

        Returns:
            A dictionary with the image path as key and a tensor of labels as
            value.
        """
        split_name = "train" if self.split == "train" else "validation"
        annotator_labels_file = os.path.join(
            self.pi_root, "labels", f"{split_name}.csv"
        )

        annotator_labels = {}

        print("Loading annotator labels...")
        with open(annotator_labels_file, "r") as f:
            for line in f:
                line = line.strip().split(",")
                annotator_labels[line[0]] = torch.tensor(
                    _to_float(line[1:]), dtype=torch.int32
                )
        return annotator_labels

    def _load_annotator_confidences(self) -> Dict[str, List[str]]:
        """Load the confidences of each annotator on their label for every image into a dictionary.

        The confidences are stored in a csv file with format:
            <image_path> <confidence_1> <confidence_2> ...

        Returns:
            A dictionary with the image path as key and a tensor of confidences
            as value.
        """

        split_name = "train" if self.split == "train" else "validation"
        annotator_confidences_file = os.path.join(
            self.pi_root, "confidences", f"{split_name}.csv"
        )

        annotator_confidences = {}

        print("Loading annotator confidences...")
        with open(annotator_confidences_file, "r") as f:
            for line in f:
                line = line.strip().split(",")
                annotator_confidences[line[0]] = torch.tensor(
                    _to_float(line[1:]), dtype=torch.float32
                )
        return annotator_confidences

    @property
    def annotator_features(self) -> torch.Tensor:
        """Returns the static features of each annotator.

        ImageNet-PI provides the model accuracy on the ImageNet validation set
        of each annotator and their log-number of parameters (normalized by
        mean and standard deviation) as additional PI features.

        Returns:
            A tensor of shape (num_annotators, 2) with the model accuracy and
            log-number of parameters of each annotator.
        """

        annotator_features_file = os.path.join(self.pi_root, "annotator_features.csv")

        annotator_features = []

        with open(annotator_features_file, "r") as f:
            for line in f:
                line = line.strip().split(",")
                annotator_features.append(
                    torch.tensor(_to_float(line), dtype=torch.float32)
                )

        return torch.stack(annotator_features)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, clean_target, confidences, annotator_labels) where
            clean_target is the index of the target class in the ImageNet
            dataset, confidences is a tensor with the confidences of each
            annotator on their label and annotator_labels is a tensor with the
            labels given by each annotator.
        """
        path, _ = self.samples[index]

        # Get file name from path
        path = os.path.basename(path)

        confidences = self.annotator_confidences[path]
        annotator_labels = self.annotator_labels[path]
        sample, clean_target = super().__getitem__(index)
        return sample, clean_target, confidences, annotator_labels
