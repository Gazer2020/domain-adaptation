import torchvision.models as models


# resnet backbone series
def get_resnet18():
    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
def get_resnet50():
    return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
def get_resnet101():
    return models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
