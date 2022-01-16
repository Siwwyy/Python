
import torch
import torchvision
import torchvision.transforms as transforms


def get_CIFAR10():
    transform_train = transforms.Compose([#transforms.RandomCrop(32, padding=4),
                                          #transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          ransforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    ds_train = torchvision.datasets.CIFAR10(root='data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform_train
    )

    ds_test = torchvision.datasets.CIFAR10(root='data', 
                                           train=False, 
                                           download=True, 
                                           transform=transform_test
    )

    return ds_train, ds_test