"""Utility functions to handle object detection."""
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json


def add_bounding_boxes(ax, bbs, category_dict=None):
    """Add bounding boxes to specified axes.

    Args:
        ax (plt.axis):
            The axis to add the bounding boxes to.
        bbs (List[Dict]):
            List of bounding boxes to display.
            Each bounding box dict has the format as specified in
            Detector.decode_output.
        category_dict (Dict):
            Map from category id to string to label bounding boxes.
            No labels if None.
    """
    for bb in bbs:
        rect = patches.Rectangle(
            (bb["x"], bb["y"]),
            bb["width"],
            bb["height"],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if category_dict is not None:
            plt.text(
                bb["x"],
                bb["y"],
                category_dict[bb["category"]]["name"],
            )

def get_category_dict(ann_file):
    category_dict = {}
    with open(ann_file, 'rb') as json_file:
        data = json.load(json_file)
        for c in data['categories']:
            category_dict[int(c['id'])] = c['name']

    return category_dict

def save_model(model, path):
    """Save model to disk.

    Args:
        model (torch.module): The model to save.
        path (str): The path to save the model to.
    """
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load model weights from disk.

    Args:
        model (torch.module): The model to load the weights into.
        path (str): The path from which to load the model weights.
        device (torch.device): The device the model weights should be on.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"], map_location=device)
    return model
