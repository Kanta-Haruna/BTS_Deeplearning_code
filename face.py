import cv2
import os

in_jpg = "/Users/harunakanta/"
out_jpg = "/Users/harunakanta/facecut"

# members = ["アールエム", "グク", "ジェイホープ", "ジミン", "シュガ", "ジン", "テテ"]
members = ["RM", "JUNGKOOK", "J-HOPE", "JIMIN", "SUGA", "JIN", "V"]
for member in members:
    image_files = os.listdir(in_jpg + member)
    for image_file in image_files:
        if image_file == ".DS_Store":
            continue
        
        image_gs = cv2.imread(in_jpg + member + "/" + image_file)
        image_gray = cv2.cvtColor(image_gs, cv2.COLOR_BGR2GRAY)
        cascade_file_path = "/Users/harunakanta/Desktop/aidemy_app/haarcascade_frontalface_alt.xml"
        cascade = cv2.CascadeClassifier(cascade_file_path)
        face = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(64, 64))
        
        if len(face) > 0:
            for rect in face:
                image_gs = image_gs[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image_gs.shape[0] < 64:
                    continue
                image_gs = cv2.resize(image_gs, (64, 64))
                save_path = out_jpg + "/" + str(image_file)
                save_image = cv2.imwrite(save_path, image_gs)
                
        else:
            print("no face")
    






