import torch
import torch.nn as nn
import torchvision.models as models


# ------------------------------------------------------------------------------------------------------------------
# CLASSES THAT MIGHT ALREADY EXIST BUT I INEXPLICABLY NEEDED TO REWRITE TO GET CLASSIFIER WORKING.
# ------------------------------------------------------------------------------------------------------------------
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d, self).__init__()
        sz = size or 1
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Normalize:
    def __init__(self, mean, std, inplace=False, dtype=torch.float):
        self.mean = torch.as_tensor(mean, dtype=dtype)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


# Pre-trained ResNet-50 artist classifier
class ArtistClassifier(nn.Sequential):
    def __init__(self, state_dict_filename, num_classes=19, device=None):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

        classifier = models.resnet50(pretrained=False)
        # Remove final layers and append the correct outputs
        modules = list(classifier.children())
        modules.pop(-1)
        modules.pop(-1)
        feature_layers = nn.Sequential(nn.Sequential(*modules))
        feature_children = list(feature_layers.children())
        # Append the layers we need (19 classes in this classifier)
        feature_children.append(nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(), nn.BatchNorm1d(4096), nn.Dropout(p=0.0), nn.Linear(4096, 512),
            nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(p=0.0), nn.Linear(512, num_classes)
        ))
        super().__init__(*feature_children)

        sd = torch.load(state_dict_filename, map_location=device)
        self.load_state_dict(sd['model'], strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.double().to(device)
