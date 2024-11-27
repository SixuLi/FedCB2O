import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple
from torch.utils import data
from torchvision import datasets, transforms
from download_emnist import EMNIST
from torchvision.transforms.functional import rotate
from torch.utils.data import Dataset

from src.utils.util import chunkify
from torch.utils.data import DataLoader

class RotatedCIFAR10:
    def __init__(self, data_path, train=True, download=True, transform=None, times_ninety_rot=0,
                 prop_to_full_dataset=1.0, seed=0):
        self.train = train
        self.transform = transform
        self.times_ninety_rot = times_ninety_rot  # Num of 90 deg rotation of the data
        self.seed = seed
        self.base_dataset = datasets.CIFAR10(root=data_path, train=self.train,
                                             transform=self.transform, download=download)
        self.targets = self.base_dataset.targets
        self.data = self.base_dataset.data
        self.prop_to_full_dataset = prop_to_full_dataset

        self.get_subset()

        if self.times_ninety_rot > 0:
            self.data = torch.rot90(torch.from_numpy(self.data),
                                    k=self.times_ninety_rot,
                                    dims=(1, 2)).numpy()

    def get_subset(self):
        random_seed = self.seed  # Same seed for same subset picking every time
        np.random.seed(random_seed)
        data_idx = np.arange(len(self.targets))
        subset_idx = np.random.choice(data_idx, int(self.prop_to_full_dataset * len(self.targets)), replace=False)
        self.data = self.data[subset_idx]
        self.targets = np.array(self.targets)[subset_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class MyCIFAR10:
    def __init__(self, args, train, download, transform,
                 times_ninety_rot=0, source_class=0,
                 prop_source_class=1.0, subset_size=200, seed=2024):
        self.args = args
        self.train = train
        self.transform = transform
        self.times_ninety_rot = times_ninety_rot # Num of 90 deg rotation of the data
        self.source_class = source_class # For example, class "plane"
        self.prop_source_class = prop_source_class
        self.subset_size = subset_size
        self.seed = seed

        # Downloads the dataset using the dummy creation of the CIFAR10 torchvision dataset.
        dummy_ds = datasets.CIFAR10(root=self.args.data_path, train=self.train,
                                    transform=self.transform, download=download)
        dummy_ds.targets = np.array(dummy_ds.targets)

        if train:
            # Create a special data split for training the models
            np.random.seed(self.seed) # Same seed for same split every time.

            all_data_idx = np.arange(0, len(dummy_ds.targets))
            source_class_idx = np.where(np.array(dummy_ds.targets) == source_class)[0]
            mask = np.ones_like(all_data_idx, dtype=bool)
            mask[source_class_idx] = False
            not_source_class_idx = all_data_idx[mask]

            source_class_images = dummy_ds.data[source_class_idx]
            source_class_labels = dummy_ds.targets[source_class_idx]
            not_source_class_images = dummy_ds.data[not_source_class_idx]
            not_source_class_labels = dummy_ds.targets[not_source_class_idx]

            source_class_data_idx = np.arange(len(source_class_labels))
            source_class_subset_idx = np.random.choice(source_class_data_idx,
                                                       int(self.prop_source_class * len(source_class_labels)),
                                                       replace=False)

            subset_source_class_images = source_class_images[source_class_subset_idx]
            subset_source_class_labels = source_class_labels[source_class_subset_idx]

            self.data = np.concatenate((not_source_class_images, subset_source_class_images), axis=0)
            self.targets = np.concatenate((not_source_class_labels, subset_source_class_labels), axis=0)

            self.get_subset()

        else:
            self.data = dummy_ds.data
            self.targets = dummy_ds.targets

        if self.times_ninety_rot > 0:
            self.data = torch.rot90(torch.from_numpy(self.data),
                                    k=self.times_ninety_rot,
                                    dims=(1,2)).numpy()

    def get_subset(self):
        # Create a subset of original dataset
        np.random.seed(self.seed) # Same seed for same subset picking every time
        data_idx = np.random.permutation(len(self.targets))
        subset_idx = np.random.choice(data_idx, self.subset_size, replace=False)
        self.data = self.data[subset_idx]
        self.targets = self.targets[subset_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class MyEMNIST:
    def __init__(self, args, train, download, transform,
                 times_ninety_rot=0, source_class=0,
                 prop_source_class=1.0, subset_size=200, seed=2024):
        self.args = args
        self.train = train
        self.transform = transform
        self.times_ninety_rot = times_ninety_rot # Num of 90 deg rotation of the data
        self.source_class = source_class # For example, digit 0
        self.prop_source_class = prop_source_class
        self.subset_size = subset_size
        self.seed = seed

        # Downloads the dataset using the dummy creation of the EMNIST torchvision dataset
        dummy_ds = EMNIST(root=self.args.data_path, split='balanced', train=self.train,
                          download=download, transform=self.transform)

        if train:
            # Create a special data split for training the models
            np.random.seed(self.seed) # Same seed for same split every time

            source_class_images = dummy_ds.data[dummy_ds.targets == source_class]
            source_class_labels = dummy_ds.targets[dummy_ds.targets == source_class]
            non_source_class_images = dummy_ds.data[(dummy_ds.targets < source_class) | (dummy_ds.targets > source_class)]
            non_source_class_labels = dummy_ds.targets[(dummy_ds.targets < source_class) | (dummy_ds.targets > source_class)]

            source_class_data_idx = np.arange(len(source_class_labels))
            source_class_subset_idx = np.random.choice(source_class_data_idx,
                                                       int(self.prop_source_class * len(source_class_labels)),
                                                       replace=False)
            subset_source_class_images = source_class_images[source_class_subset_idx]
            subset_source_class_labels = source_class_labels[source_class_subset_idx]

            self.data = torch.concat((non_source_class_images, subset_source_class_images), axis=0)
            self.targets = torch.concat((non_source_class_labels, subset_source_class_labels), axis=0)

            self.get_subset()

        else:
            self.data = dummy_ds.data
            self.targets = dummy_ds.targets

        if self.times_ninety_rot > 0:
            self.data = torch.rot90(self.data,
                                    k=self.times_ninety_rot,
                                    dims=(1,2))
        # normalize data to have 0 ~ 1 range in each pixel
        self.data = self.data / 255.0
        self.data = self.data.reshape(-1, 1, 28, 28)

    def get_subset(self):
        # Create a subset of original dataset
        np.random.seed(self.seed) # Same seedd for same subset picking every time
        data_idx = np.arange(len(self.targets))
        subset_idx = np.random.choice(data_idx, self.subset_size, replace=False)
        self.data = self.data[subset_idx]
        self.targets = self.targets[subset_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        # img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class RotatedEMNIST:
    def __init__(self, data_path, train=True, download=True, transform=None, times_ninety_rot=0,
                 prop_to_full_dataset=1.0, seed=0):
        self.train = train
        self.transform = transform
        self.times_ninety_rot = times_ninety_rot  # Num of 90 deg rotation of the data
        self.seed = seed
        self.base_dataset = EMNIST(root=data_path, split='balanced', train=self.train,
                          download=download, transform=self.transform)
        self.targets = self.base_dataset.targets
        self.data = self.base_dataset.data
        self.subset_size = int(len(self.targets) * prop_to_full_dataset)
        self.prop_to_full_dataset = prop_to_full_dataset

        self.get_subset()

        if self.times_ninety_rot > 0:
            self.data = torch.rot90(self.data,
                                    k=self.times_ninety_rot,
                                    dims=(1, 2))

    def get_subset(self):
        # Create a subset of original dataset
        np.random.seed(self.seed)  # Same seedd for same subset picking every time
        data_idx = np.arange(len(self.targets))
        subset_idx = np.random.choice(data_idx, self.subset_size, replace=False)
        self.data = self.data[subset_idx]
        self.targets = self.targets[subset_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        # img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class CustomDataset:
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset1 = RotatedEMNIST(data_path='../data', train=True, download=True, transform=transform, times_ninety_rot=0)
    trainset2 = RotatedEMNIST(data_path='../data', train=True, download=True, transform=transform, times_ninety_rot=2)
    # target_images_list = [389, 561, 874, 1605, 3378, 3678, 4528, 9744, 19165, 19500, 21422, 22984, 32941,
    #                       34287, 34385, 36005, 37365, 37533, 38658, 38735, 39824, 40138, 41336, 41861,
    #                       47001, 47026, 48003, 48030, 49163, 49588]
    trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=4, shuffle=True)
    trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=4, shuffle=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    #
    # # Class labels in CIFAR-10
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # # Function to convert a tensor to an image
    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.imshow(npimg, cmap='Greys')
        plt.axis('off')
        plt.show()

    # get some random training images
    dataiter1 = iter(trainloader1)
    images1, labels1 = next(dataiter1)

    dataiter2 = iter(trainloader2)
    images2, labels2 = next(dataiter2)

    images = torch.concat([images1, images2])

    # show images
    # imshow(torchvision.utils.make_grid(images, nrow=4, padding=0))
    plt.figure(figsize=(12, 8))
    n_images = 8
    rand_imgs1, labels1 = next(iter(trainloader1))
    rand_imgs2, labels2 = next(iter(trainloader2))
    rand_imgs = torch.concat([rand_imgs1, rand_imgs2])
    ints = range(8)

    fig, ax_arr = plt.subplots(2,4)
    ax_arr = ax_arr.flatten()

    for n, ix in enumerate(ints):
        img = rand_imgs[ix]
        ax_arr[n].imshow(img[0].detach().cpu().numpy().T)
        ax_arr[n].axis('off')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    # plt.tight_layout()
    plt.savefig('../results/visualization/rotated_emnist.png', bbox_inches='tight')
    # plt.savefig(os.path.join(self.train_init.output_path, 'dataset_visual.png'))
    plt.show()

    # # Visualize the images with the specified indices
    # for i, data in enumerate(trainloader, 0):
    #     # if i in target_images_list:
    #     image, label = data
    #     imshow(image.squeeze())