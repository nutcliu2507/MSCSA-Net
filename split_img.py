# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:22:08 2021

@author: GREEN&LYC
"""

import cv2
import os
import numpy as np
import random

if __name__ == "__main__":

    def random_crop(image1, image2, size=512):

        #        image = resize_image(image)

        h, w = image1.shape[:2]

        y = np.random.randint(0, h - size)
        x = np.random.randint(0, w - size)

        data = image1[y:y + size, x:x + size, :]
        gt = image2[y:y + size, x:x + size, :]

        rr = np.random.randint(0, 2)
        if rr == 0:
            data = cv2.flip(data, 0)
            gt = cv2.flip(gt, 0)
            return data, gt
        elif rr == 1:
            #            img_list.append(imgl0)
            data = cv2.flip(data, 1)
            gt = cv2.flip(gt, 1)
            return data, gt
        else:
            #            img_list.append(imgl1)
            return data, gt


    datapath = r"./data3/image/"  # change this dirpath.
    listdir = os.listdir(datapath)
    gtpath = r"./data3/gt/"  # change this dirpath.
    #    listdir = os.listdir(gtpath)

    dnewdir = os.path.join(datapath, 'split2')
    gnewdir = os.path.join(gtpath, 'split2')  # make a new dir in dirpath.
    #    print(newdir)

    if (os.path.exists(dnewdir) == False):
        os.mkdir(dnewdir)
    if (os.path.exists(gnewdir) == False):
        os.mkdir(gnewdir)

    for i in listdir:
        if os.path.isdir(os.path.join(datapath, i)):
            continue
        #        elif os.path.isdir(os.path.join(gtpath, i)):
        #            continue
        #        if os.path.isfile(os.path.join(path, i)):
        # if i.split('.')[1] == "tif":
        datafilepath = os.path.join(datapath, i)
        gtfilepath = os.path.join(gtpath, i)
        filename = i.split('.')[0]

        img = cv2.imread(datafilepath)
        gtimg = cv2.imread(gtfilepath)
        count = 1
        #        [h, w] = imgg.shape[:2]
        #        img = cv2.resize(imgg,(1600,1600))
        img_list = []
        gtimg_list = []
        # img_path=[]
        #        for j in range(5):
        #            for k in range(5):
        #
        #                #i[j] =
        #                #rightpath = os.path.join(newdir, filename) + "_gt_trian.jpg"
        #
        #                #print(filepath, (h, w))
        #                #imgl = img[j*int(h/5):(j+1)*int(h/5), k*int(w/5):(k+1)*int(w/5), :]
        #                imgl = img[j*320:(j+1)*320, k*320:(k+1)*320, :]
        #                img_list.append(imgl)
        #
        #                imgl0 = cv2.flip(imgl,0)
        #                img_list.append(imgl0)
        #                imgl1 = cv2.flip(imgl,1)
        #                img_list.append(imgl1)
        #                imgl2 = cv2.flip(imgl,-1)
        #                img_list.append(imgl2)
        #
        #                imgl3=cv2.rotate(imgl,cv2.ROTATE_90_CLOCKWISE)
        #                img_list.append(imgl3)
        #                imgl4=cv2.rotate(imgl,cv2.ROTATE_90_COUNTERCLOCKWISE)
        #                img_list.append(imgl4)

        while count <= 150:
            imgg, gtimgg = random_crop(img, gtimg)

            img_path = os.path.join(dnewdir, filename) + "_" + str(count) + ".jpg"
            gtimg_path = os.path.join(gnewdir, filename) + "_" + str(count) + ".jpg"
            cv2.imwrite(img_path, imgg)
            cv2.imwrite(gtimg_path, gtimgg)
            count += 1

#        for _ in range(150):   
#            img_path = os.path.join(newdir, filename) + "_" + str(_+1)+ ".jpg"
#            cv2.imwrite (img_path,img_list[_])

# cv2.imwrite(leftpath, img[j])
# cv2.imwrite(rightpath, rimg)
