import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Function

from copy import deepcopy


__all__ = ['ReverseLayerF','dann_AlexNet', 'dann_alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class dann_AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(dann_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.discriminator = nn.Sequential()


    def forward(self, x, alpha=None):
        x = self.features(x)
        x = self.avgpool(x)
        # Flatten the features:
        x = torch.flatten(x,1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reversed_x = ReverseLayerF.apply(x, alpha)
            discriminator_output = self.discriminator(reversed_x)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # perform classification
            class_outputs = self.classifier(x)
            return class_outputs

def dann_alexnet(pretrained=False, progress=True, num_classes=7, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = dann_AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    
    # Create the discriminator branch by coping the classifier
    model.discriminator = deepcopy(model.classifier)
    
    # Adjust the last layers to be consistent with the task
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.dicriminator = nn.Linear(4096, 2)

    return model
