from typing import Union

import torch
import torch.nn as nn


TensorType = torch.tensor
""" 
    Possible Tensor type
"""

ShapeType = Union[tuple, torch.Size]
""" 
    Possible Shape types of the tensor
"""

class Model_First(nn.Module):
    """
    First and base model of our project (derives from Model_Base, look at model_base.py)

    Attributes
    ----------
    name : str
        name of model
    input_shape : ShapeType (look at possible shape types in config.py file)
        input shape of tensors
    num_classes : int
        number of classes in prediction
    **kwargs : Any
        keyword arguments (currently not used)

    Methods
    -------
    forward(self, x: TensorType = None) -> TensorType:
        Forward propagation's method of model
    """

    def __init__(
        self,
        name: str = "Model_First",
        input_shape: ShapeType = (1, 3, 64, 64),
        num_classes: int = 2,
        **kwargs
    ):
        super().__init__(name, input_shape, num_classes)

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # conv3 1x1 convolution as a Fully Connected layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        # conv4 1x1 convolution as a Fully Connected layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
        )

    def forward(self, x: TensorType = None) -> TensorType:

        # Input convolution
        output = self.conv1(x)

        # Second convolution
        output = self.conv2(output)

        # Third convolution, 1x1 -> Flatten Conv
        output = self.conv3(output)

        # Fourth convolution, 1x1 -> Flatten Conv
        output = self.conv4(output)
        return output
