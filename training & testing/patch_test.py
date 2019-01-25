import numpy as np
import os
import handlers.args as args
import handlers.load as load
import cv2 as cv
# from handlers.proc import MyProcess as proc
# from scipy import misc
arg = args.get_args()
cityName = arg.city_name
arg.input_folder = '../data/all_patches/'
arg.output_folder = '../results/multi_city'
arg.no_depth = True

#----Load Data---
print("Loading Images: "+arg.input_folder+"rgb/")
rgb, rgbFileNames = load.load_data(arg.input_folder+"rgb/", returnNames = True)

print("Loaded "+ str(len(rgb)) + " images")

x = rgb

del rgb

#----KERAS ENV-------
os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.gpu)

from keras.models import Model, load_model

print("Loading Model")
model = load_model(arg.output_folder+'/model.hdf5', compile=False)

train_args = load.get_args(arg.output_folder)
arg.mean = np.array(train_args['mean'])

#_!_!_!_!_!_!_RUN NETWORK_!_!_!_!_!_
print("Running Network")

def aug(img):
	img = cv.resize(img,(300, 300), interpolation = cv.INTER_CUBIC)
	img1 = img[0:200, 0:200, :]
	img2 = img[100:300, 0:200, :]
	img3 = img[0:200, 100:300, :]
	img4 = img[100:300, 100:300, :]
	return img1, img2, img3, img4


for i in range(len(rgbFileNames)):
    rgbImg = x[i]
    c1, c2, c3 = cv.split(np.ubyte(rgbImg))
    rgbImg = cv.merge((cv.equalizeHist(c1), cv.equalizeHist(c2), cv.equalizeHist(c3)))
    rgbImg = cv.resize(rgbImg,(200, 200), interpolation = cv.INTER_CUBIC)
    rgbImg = rgbImg - arg.mean
    # img1, img2, img3, img4 = aug(rgbImg)

    predImg = model.predict(np.reshape(rgbImg, (1,)+rgbImg.shape))
    rgbImg = predImg[0]

    # predImg = model.predict(np.reshape(img2, (1,)+img2.shape))
    # img2 = predImg[0]

    # predImg = model.predict(np.reshape(img3, (1,)+img3.shape))
    # img3 = predImg[0]

    # predImg = model.predict(np.reshape(img4, (1,)+img4.shape))
    # img4 = predImg[0]

    # resImg = np.zeros((300, 300, 1))
    # resImg[0:200, 0:200, :] += img1
    # resImg[100:300, 0:200, :] += img2
    # resImg[0:200, 100:300, :] += img3
    # resImg[100:300, 100:300, :] += img4

    # resImg[100:200, 100:200, :] /= 2

    # img = cv.resize(resImg,(200, 200), interpolation = cv.INTER_CUBIC)

    cv.imwrite(arg.input_folder + "pred/" + rgbFileNames[i], rgbImg)
