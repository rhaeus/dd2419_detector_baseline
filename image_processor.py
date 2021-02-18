#!/usr/bin/env python

import utils
from torchvision import models
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from detector import Detector
import torchvision.transforms.functional as TF


class ImageProcessor():
    def __init__(self, model_path, ann_path):
        self.device = torch.device('cpu')

        # Init model
        self.detector = Detector().to(self.device)

        # load a trained model
        self.detector = utils.load_model(self.detector, model_path, self.device)

        # load category dictionary from annotation file
        self.category_dict = utils.get_category_dict(ann_path)

    def detect_and_classify(self, image, threshold):
        """ Gets an image as PIL image
        and performs detection and classification
        Returns:
            List[List[Dict]]
            List containing a list of detected bounding boxes in each image.
            Each dictionary contains the following keys:
                - "x": Top-left corner column
                - "y": Top-left corner row
                - "width": Width of bounding box in pixel
                - "height": Height of bounding box in pixel
                - "category": Category (not implemented yet!)
         """

        
        # torch_image = TF.to_tensor(image)
        # torch_image = torch.stack(torch_image)
        # torch_image = torch_image.to(self.device)

        # for some reason it needs to be torch.stacked
        # and therefore it needs to be a list even though it is only one image
        torch_image = []
        torch_image.append(TF.to_tensor(image))

        if torch_image:
            torch_image = torch.stack(torch_image)
            torch_image = torch_image.to(self.device)


        with torch.no_grad():
            out = self.detector(torch_image).cpu()
            bbs = self.detector.decode_output(out, threshold)
            return bbs


    def overlay_image(self, pil_image, bbs):
        """ Gets a PIL image and the bounding boxes
        Returns a PIL image with bounding box and category overlay
        """

        return utils.add_bounding_boxes_pil(pil_image, bbs, self.category_dict)


