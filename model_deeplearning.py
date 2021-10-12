import cv2
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D, Input
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

members = ["RM", "JUNGKOOK", "J-HOPE", "JIMIN", "SUGA", "JIN", "V"]
train_jpg = "/Users/harunakanta/BTS/TRAIN/"
test_jpg = "/Users/harunakanta/BTS/TEST/"


X_train = []
y_train = []
for i, member in enumerate(members):
    image_files = os.listdir(train_jpg + member)
    for image in image_files:
        if image == ".DS_Store":
            continue
        img = cv2.imread(train_jpg + member + "/" + image)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_train.append(img)
        y_train.append(i)


X_test = []
y_test = []
for i, member in enumerate(members):
    image_files = os.listdir(test_jpg + member)
    for image in image_files:
        if image == ".DS_Store":
            continue
        img = cv2.imread(test_jpg + member + "/" + image)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        X_test.append(img)
        y_test.append(i)


X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
index = 0

kf = KFold(n_splits=5, shuffle=True)
for train_index, eval_index in kf.split(X_train,y_train):
    X_tra, X_eval = X_train[train_index], X_train[eval_index]
    y_tra, y_eval = y_train[train_index], y_train[eval_index]
    
    model_weights = "/Users/harunakanta/btsapp[%d].h5" % index
    index = index+1

    model = Sequential()
    model.add(Conv2D(input_shape=(150,150,3), filters=32, kernel_size=(3,3),strides=(1,1), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1,1), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=32, kernel_size=(3,3),strides=(1,1), padding="same"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation("softmax"))



    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    checkpointer = ModelCheckpoint(model_weights, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)
    history = model.fit(X_train, y_train,
                        batch_size=128,
                        epochs=150,
                        verbose=1,
                        validation_data=(X_eval, y_eval),
                        callbacks=[early_stopping, checkpointer])
    score = model.evaluate(X_test,  y_test, batch_size=128, verbose=0)
    print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))
    
    pred = np.argmax(model.predict(X_test[0:10]), axis=1)
    y_pred = model.predict_classes(X_test)

    plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
    plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.show()

