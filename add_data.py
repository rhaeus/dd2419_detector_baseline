#!/usr/bin/env python

import numpy as np
import utils
import PIL
import json

from PIL import Image

def main():

    annFile="./dd2419_coco/annotations/training.json"
    with open(annFile) as json_file: 
        data_final = json.load(json_file)

    newFile="./240_coco_imglab.json"
    with open(newFile) as json_file: 
        data_new = json.load(json_file)

    ind = data_final["images"][-1]["id"]

    for image in range(0,len(data_new["images"])):
        data_final["images"].append(data_new["images"][image])
        data_final["images"][-1]["id"]+=ind

    for annot in range(0, len(data_new["annotations"])):
        model = data_final["annotations"][0].copy()
        model["image_id"]=data_new["annotations"][annot]["image_id"]+ind
        model["id"]=model["image_id"]
        if data_new["annotations"][annot]["category_id"]==2:
            model["category_id"]=3
        elif data_new["annotations"][annot]["category_id"]==3:
            model["category_id"]=4
        elif data_new["annotations"][annot]["category_id"]==4:
            model["category_id"]=5
        elif data_new["annotations"][annot]["category_id"]==5:
            model["category_id"]=6
        elif data_new["annotations"][annot]["category_id"]==6:
            model["category_id"]=0
        elif data_new["annotations"][annot]["category_id"]==7:
            model["category_id"]=8
        elif data_new["annotations"][annot]["category_id"]==8:
            model["category_id"]=9
        elif data_new["annotations"][annot]["category_id"]==9:
            model["category_id"]=14
        else :
            model["category_id"]=data_new["annotations"][annot]["category_id"]
        model["bbox"]=data_new["annotations"][annot]["bbox"]
        data_final["annotations"].append(model)



            

    #save new .json annotation file
    with open("./dd2419_coco/annotations/newdata.json", "w") as outfile:  
        json.dump(data_final, outfile) 

    
    #im = np.array(Image.open('./dd2419_coco/training/000000.jpg').convert('L'))
    #gr_im= Image.fromarray(im).save('gr_000000.jpg')



    #img = Image.open('./dd2419_coco/training/000000.jpg')
    #flip_1 = np.fliplr(img)
    # TensorFlow. 'x' = A placeholder for an image.
    height = 480.0
    width = 640.0
    #shape = [height, width, channels]
    #x = tf.placeholder(dtype = tf.float32, shape = shape)
    #flip_2 = tf.image.flip_up_down(x)
    #flip_3 = tf.image.flip_left_right(x)
    #flip_4 = tf.image.random_flip_up_down(x)
    #flip_5 = tf.image.random_flip_left_right(x)

    #utils.save(flip_1, './augm/000000.jpg')








if __name__ == "__main__":
    main()