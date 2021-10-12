# coding:utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def scratch_imag(img,rotateclock=False, rotate180=False, thr=False, filt=False, resize=True):
    # 水増し手法
    # flip=画像の反転
    # thr=閾値処理
    # filt=ぼかし
    methods = [rotateclock,rotate180,thr,filt,resize]
    
    # オリジナルを格納
    images = [img]
    
    scratch = np.array([
        lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        lambda x: cv2.rotate(x, cv2.ROTATE_180),
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        lambda x: cv2.GaussianBlur(x, (5,5), 0),
        lambda x: cv2.resize(x,(150, 150))
    ])
    doubling_img = lambda f, imag: (imag + [f(i) for i in imag])
    for func in scratch[methods]:
        images = doubling_img(func, images)
    return images

members = ["RM", "JUNGKOOK", "J-HOPE", "JIMIN", "SUGA", "JIN", "V"]
# members = ["アールエム", "グク", "ジェイホープ", "ジミン", "シュガ", "ジン", "テテ"]
# 画像の読み込み
for member in members:
    in_jpg = "/Users/harunakanta/BTS/TEST/" + member
    image_files = os.listdir(in_jpg)

    for image in image_files:
        if image == ".DS_Store":
            continue
        img = cv2.imread("/Users/harunakanta/BTS/TEST/" + member + "/" + image)
        
        
        # 画像の水増し
        scratch_images = scratch_imag(img)

        for num, im in enumerate(scratch_images):
            out_jpg = "/Users/harunakanta/BTS/TEST/"
            save_path = out_jpg + "/" + image + ".jpeg"
            save_image = cv2.imwrite(save_path,im)
