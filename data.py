'''
Reference taken from the Dassl.pytorch repo: https://github.com/KaiyangZhou/Dassl.pytorch
'''
import os
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from collections import defaultdict
from PIL import Image
from prettytable import PrettyTable


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError("No file exists at {}".format(path))

    while True:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except IOError:
            print(
                "Cannot read image from {}, "
                "probably due to heavy IO. Will re-try".format(path)
            )


class Datum:

    def __init__(self, impath="", label=0, classname=""):

        assert isinstance(impath, str)
        self._impath = impath
        self._label = label
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def classname(self):
        return self._classname


class DatasetBase:

    def __init__(self, train=None, val=None):

        self._train = train  # labeled training data
        self._val = val  # labeled validation data

        self._num_classes = self.get_num_classes(train)
        self._lab2cname, self._classnames = self.get_lab2cname(train)
        self._class_count = self.get_class_count(train)
        self._weight_class = self.get_weight_class(train, self.class_count)

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def class_distribution(self):
        table = PrettyTable()
        table.add_column("Class", self.classnames)
        table.add_column("Count", list(self.class_count.values()))
        table.add_column("Weights Per Class", ['%.5f' % (1. / val) for val in list(self.class_count.values())])
        return table

    @property
    def weight_class(self):
        return self._weight_class

    @property
    def class_count(self):
        return self._class_count

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_weight_class(self, data_source, class_count):
        """Count the weight per sample.

        Args:
            class_count (dict): a dict of class count.
            data_source (list): a list of item Datum.
        """
        weight_class = list()
        for item in data_source:
            weight_class.append(1. / class_count[item.label])
        return np.array(weight_class)

    def get_class_count(self, data_source):
        """Count number of samples per class.

        Args:
            data_source (list): a list of Datum objects.
        """
        class_count = defaultdict(int)
        for item in data_source:
            class_count[item.label] += 1
        return class_count

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames


class DataList(DatasetBase):

    '''
    Create a list of Datum Objects for every image and class.

    Args:
        rootpath: Path to the dataset. Must consist of [train, {val, test}] folders
    '''

    def __init__(self, rootpath, percent=1.0):

        self.dataset_dir = rootpath
        self.percent = percent
        train = self._read_data(split='train')
        val = self._read_data(split='val')

        super().__init__(train=train, val=val)

    def _read_data(self, split='train'):

        items = list()

        parent_dir = osp.join(self.dataset_dir, split)
        class_names = listdir_nohidden(parent_dir)
        class_names.sort()

        for label, class_name in enumerate(class_names):
            class_path = osp.join(parent_dir, class_name)
            imnames = listdir_nohidden(class_path)
            if split == 'train':
                imnames = imnames[:round(len(imnames) * self.percent)]
            for imname in imnames:
                impath = osp.join(class_path, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=class_name
                )
                items.append(item)

        return items


class MyDataset(Dataset):

    def __init__(self, data_source, train_transform=None, test_transform=None, is_train=True):

        self.data_source = data_source

        assert (train_transform is not None or test_transform is not None)  # One transform should be compulsory
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.is_train = is_train

    def __len__(self):

        return len(self.data_source)

    def __getitem__(self, idx):

        item = self.data_source[idx]

        output = {
            "label": item.label,
        }

        img0 = read_image(item.impath)

        if self.is_train:
            img = self.train_transform(img0)
        else:
            img = self.test_transform(img0)

        output['img'] = img

        return output
