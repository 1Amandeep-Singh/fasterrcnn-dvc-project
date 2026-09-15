"""Builds a torchvision Faster R-CNN model with a classification head sized
to the dataset's actual number of classes (background + N objects)."""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, pretrained_backbone: bool = True):
    """
    num_classes must include the background class, e.g. for 3 object types
    pass num_classes=4.
    """
    weights = "DEFAULT" if pretrained_backbone else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Swap the box-predictor head so its output matches our class count
    # instead of COCO's default 91 classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
