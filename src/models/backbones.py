import torchvision.models as models


# resnet backbone series
resnet18_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet50_backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet101_backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
