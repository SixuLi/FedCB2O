import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import RotatedCIFAR10


if __name__ == '__main__':
    # Define a transform to convert the images to tensor and normalize them
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the CIFAR-10 training set
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    # CIFAR-10 classes
    classes = trainset.classes

    # Define a function to show images
    def imshow(img, label='cat'):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')  # Remove axes
        if label == 'dog':
            plt.savefig('../results/images_dogs_inference.png', bbox_inches='tight')
        elif label == 'cat':
            plt.savefig('../results/images_cats_inference.png', bbox_inches='tight')
        plt.show()

    # Find cat images (class 3)
    cat_images = []
    for i in range(2000, len(trainset)):
        if trainset[i][1] == 3:  # Class 3 corresponds to 'cat'
            cat_images.append(trainset[i][0])
        if len(cat_images) == 2:  # We need only 5 images
            break

    # Show the 5 cat images
    imshow(torchvision.utils.make_grid(cat_images, nrow=1, padding=4, pad_value=1), label='cat')

    # Find dog images (class 5)
    dog_images = []
    for i in range(800, len(trainset)):
        if trainset[i][1] == 5:  # Class 3 corresponds to 'cat'
            dog_images.append(trainset[i][0])
        if len(dog_images) == 2:  # We need only 5 images
            break

    # Show the 5 cat images
    imshow(torchvision.utils.make_grid(dog_images, nrow=1, padding=4, pad_value=1), label='dog')



