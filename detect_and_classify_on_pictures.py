#!/usr/bin/env python

import utils
from torchvision import models
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from detector import Detector
import torchvision.transforms.functional as TF
import math

def main():
    device = torch.device('cpu')

    # Init model
    detector = Detector().to(device)

    # load a trained model
    detector = utils.load_model(detector, './trained_models/weight_2_1_2_best.pt', device)

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

            #keep only the best bbox
            if len(bbs[i])>1:
                
                bbx=[bbs[i][0]]
                for elem in bbs[i][1:]:
                    
                    i = False
                    for box in bbx:
                        #print('hehe')
                        x_center_elem = elem['x'].item()-elem['width'].item()/2
                        x_center_box = box['x'].item()-box['width'].item()/2
                        y_center_elem = elem['y'].item()-elem['height'].item()/2
                        y_center_box = box['y'].item()-box['height'].item()/2
                        if math.sqrt((y_center_elem-y_center_box)**2+(x_center_elem-x_center_box)**2) <100:
                            if box['category_conf']<elem['category_conf']:
                                bbx.remove(box)
                                bbx.append(elem)
                            i=True
                    if i == False:
                        bbx.append(elem)
                                
            #        
            #    bbx = [max(bbs[i], key=lambda x:x['category_conf'])]
            #    #print(bbx)
            else:
                bbx = bbs[i][:]
            # add bounding boxes
            
            for elem in bbx :
                print(elem['category_conf'])
                if elem['category_conf']<0.8:
                    print(elem['category_conf'])
                    bbx.remove(elem)

            utils.add_bounding_boxes(ax, bbx, category_dict)
            

            # wandb.log(
            #     {"test_img_{i}".format(i=i): figure}, step=current_iteration
            # )
            plt.show()
            plt.close()
    

if __name__ == "__main__":
    main()