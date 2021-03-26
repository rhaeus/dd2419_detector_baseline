#!/usr/bin/env python

import numpy as np
import utils
import PIL
import json

from PIL import Image

def main():

    annFile="./dd2419_coco/annotations/newdata.json"
    with open(annFile) as json_file: 
        data1 = json.load(json_file)

    index = data1["images"][-1]["id"]
    for image in range(0,len(data1["annotations"])):
        data = data1.copy()
        #get image path
        img_id = data["annotations"][image]["image_id"]
        imgFile = data["images"][img_id]['file_name']
        imgFile_path = './dd2419_coco/training/'+imgFile
        #print(image)
        img_o = Image.open(imgFile_path)

        #get information for cropping
        x = data["annotations"][image]["bbox"][0]
        y = data["annotations"][image]["bbox"][1]
        w = data["annotations"][image]["bbox"][2]
        h = data["annotations"][image]["bbox"][3]
        x_center = x + w / 2.0
        y_center = y + h / 2.0

        #crop images
        upper = y_center - 100
        left = x_center - 133
        lower = y_center + 100
        right = x_center + 133
        img = img_o.crop((left, upper, right, lower))
        img = img.resize((640,480))

        #black and white converting
        #im = np.array(Image.open(imgFile_path).convert('L'))

        #new dict for image info
        nb=index+image
        imgdict = {"id":nb , "width": 640, "height": 480, "file_name": "cr_"+imgFile}

        #add it in data
        data["images"].append(imgdict)

        #new dict for bbox info
        bboxdict = data["annotations"][image].copy()
        bboxdict["id"]=nb
        bboxdict["image_id"]=nb
        bboxdict["bbox"]=[320 - w/2.0, 240 - h/2.0, w, h]

        #add it in data
        data["annotations"].append(bboxdict)

        #change info in data
        data['info']["version"]=2.0
        data['info']["description"]='DD2419 traffic sign dataset augmented with cropped images'

        #save new file 
        #gr_im= Image.fromarray(im).save("./dd2419_coco/training/tr_"+imgFile)
        img.save("./dd2419_coco/training/cr_"+imgFile)

        #get info for translation
        if x_center + w/2 < 540:
            #translation
            a = 1
            b = 0   
            c = -100 #left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = 0 #up/down (i.e. 5/-5)
            img1 = img_o.transform(img_o.size, Image.AFFINE, [a, b, c, d, e, f], resample=Image.BILINEAR, fillcolor=(255,255,255))

            #new dict for image info
            nb1=2*index+image
            imgdict = {"id":nb1 , "width": 640, "height": 480, "file_name": "tr_"+imgFile}

            #add it in data
            data["images"].append(imgdict)

             #new dict for bbox info
            bboxdict = data["annotations"][image].copy()
            bboxdict["id"]=nb1
            bboxdict["image_id"]=nb1
            bboxdict["bbox"]=[x+100, y, w, h]

            #add it in data
            data["annotations"].append(bboxdict)

            #save new file
            img1.save("./dd2419_coco/training/tr_"+imgFile)

        #get info for translation
        if ((y_center + h/2 < 380) and (x_center + w/2 < 540)):
            #translation
            a = 1
            b = 0   
            c = -100 #left/right (i.e. 5/-5)
            d = 0
            e = 1
            f = -100 #up/down (i.e. 5/-5)
            img2 = img_o.transform(img_o.size, Image.AFFINE, [a, b, c, d, e, f], resample=Image.BILINEAR, fillcolor=(0,0,0))

            #new dict for image info
            nb2=3*index+image
            imgdict = {"id":nb2 , "width": 640, "height": 480, "file_name": "tr1_"+imgFile}

            #add it in data
            data["images"].append(imgdict)

             #new dict for bbox info
            bboxdict = data["annotations"][image].copy()
            bboxdict["id"]=nb2
            bboxdict["image_id"]=nb2
            bboxdict["bbox"]=[x+100, y+100, w, h]

            #add it in data
            data["annotations"].append(bboxdict)

            #save new file
            img2.save("./dd2419_coco/training/tr1_"+imgFile)
            

    #save new .json annotation file
    with open("./dd2419_coco/annotations/augmented.json", "w") as outfile:  
        json.dump(data, outfile) 

    
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