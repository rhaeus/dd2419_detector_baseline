#!/usr/bin/env python

import utils
from torchvision import models
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from detector import Detector
import torchvision.transforms.functional as TF

def main():
    device = torch.device('cpu')

    # Init model
    detector = Detector().to(device)

    # load a trained model
    detector = utils.load_model(detector, './trained_models/det_2021-02-16_15-14-19-181578.pt', device)

    category_dict = utils.get_category_dict('./dd2419_coco/annotations/training.json')
    

    # load test images
    test_images = []
    directory = "./test_images"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(directory, file_name)
            test_image = Image.open(file_path)
            test_images.append(TF.to_tensor(test_image))

    if test_images:
        test_images = torch.stack(test_images)
        test_images = test_images.to(device)

    with torch.no_grad():
        out = detector(test_images).cpu()
        bbs = detector.decode_output(out, 0.5)

        for i, test_image in enumerate(test_images):
            figure, ax = plt.subplots(1)
            plt.imshow(test_image.cpu().permute(1, 2, 0))
            plt.imshow(
                out[i, 4, :, :],
                interpolation="nearest",
                extent=(0, 640, 480, 0),
                alpha=0.7,
            )

            # add bounding boxes
            utils.add_bounding_boxes(ax, bbs[i], category_dict)

            # wandb.log(
            #     {"test_img_{i}".format(i=i): figure}, step=current_iteration
            # )
            plt.show()
            plt.close()
    

if __name__ == "__main__":
    main()