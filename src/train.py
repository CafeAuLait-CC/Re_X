import os
import sys
import numpy as np
import argparse
from scipy import misc

ar = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ar.add_argument('gpu', type = int, help = "Please enter GPU to use")
ar.add_argument('name', type = str, help = "Please training name - for save folder")
ar.add_argument('model_name', type = str, help = "Please enter model type - unet,mynet,resnet50")
ar.add_argument('patch_size', type = int, help = "Please enter model patch size")
ar.add_argument('-o','--patch_size_out', type = int, help = "Please enter model output patch size")
ar.add_argument('-b','--batch_size', type = int, help = "Please enter model batch size")
ar.add_argument('--predef', action='store_true', help='Option defining whether to use predefined patches')
ar.add_argument('--load_balance', action='store_true', help='Option defining whether have load balancing')
ar.add_argument('--continue_model', action='store_true', help='Option defining whether to continue existing model')
ar.add_argument('--dilate', action='store_true', help='Transfer weights into dilation convolutions')
args = ar.parse_args()
device = 'cuda'+str(args.gpu)
name = args.name
model_name = args.model_name
patch_size = (args.patch_size,args.patch_size)
predef = args.predef
cont = args.continue_model
single_pixel_out = False
# label = args.labels


batch_size = args.batch_size
if batch_size == None:
    batch_size = 32
name += "_p" + str(args.patch_size) + "_b"+str(batch_size)   
dilate = args.dilate
#----Image Paths---

BASE_PATH = '../data/'
REAL_PATH = BASE_PATH+'real/'
OUT_PATH = BASE_PATH+'out/'
     
TRAIN_PATH = OUT_PATH+name
if not cont:
    if os.path.exists(TRAIN_PATH):
        i = 0
        while os.path.exists(TRAIN_PATH):
            TRAIN_PATH = OUT_PATH+name+'_'+str(i)
            i += 1
    os.makedirs(TRAIN_PATH)
else:
    if not os.path.exists(TRAIN_PATH):
        print("Model does not exist")
        sys.exit(0)



#----KERAS ENV-------

os.environ["THEANO_FLAGS"]='device='+device
sys.setrecursionlimit(50000)

import load,proc,math,plotting,model
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model


#----Load Data---
if not predef:
    set_imgs= load.load_all(REAL_PATH)#loads [y,rgb,d]
    set_imgs2= load.load_all(REAL_PATH,comp=True)#loads [y_ng,rgb_ng,d_ng]
    print("Loading Done")
    print("Preprocessing")
    x_img = set_imgs[1]
    y_img = set_imgs[0]#proc.preprocess_ys(set_imgs[0],x_img,labels=label)
    x_img = proc._join(x_img,set_imgs[2:])
    #x_img = proc._join(x_img,set_defer[0:1],depth=3)
    x_img_val = set_imgs2[1]
    y_img_val = set_imgs2[0]#proc.preprocess_ys(set_imgs2[0],x_img_val,labels=label)
    x_img_val = proc._join(x_img_val,set_imgs2[2:]) 
    #x_img_val = proc._join(x_img_val,set_defer2[0:1],depth=3)
    
    #hnv_imgs = load.load_data_simp([REAL_PATH+'hnv',REAL_PATH+'hnv_ng'])
    #x_img = proc._join(x_img,hnv_imgs[0:1],depth=2)
    #x_img_val = proc._join(x_img_val,hnv_imgs[1:2],depth=2)

print("Testing: X: "+str(len(x_img))+" of depth: "+str(x_img[0].shape[2]))
print("Test To: Y: "+str(len(y_img))+" of depth: "+str(y_img[0].shape[2]))

if not cont or dilate:
    print("Generating Model")
    model = model.gen((patch_size+(x_img[0].shape[2],)),model_name,nclasses=None)
    print("Compiling Model")
    #plot_model(model, to_file=TRAIN_PATH+'/model.png',show_shapes=True,show_layer_names=True)
    model.compile('RMSprop', 'mean_squared_error')
else:
    print("Loading Model")
    model = load_model(TRAIN_PATH+'/model.hdf5')


checkpointer = ModelCheckpoint(filepath=TRAIN_PATH+'/model.hdf5', verbose=0, save_best_only=False)
breakPateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
#plotter = plotting.PlotLoss(TRAIN_PATH, x_img, y_img, x_img_val, y_img_val, patch_size, patch_out=patch_size_out, labels=label)
# plotter = plotting.PlotLoss(TRAIN_PATH, None, None, None, None, None, plot_images=False)
plotter = plotting.PlotLoss(TRAIN_PATH, x_img, y_img, x_img_val, y_img_val, patch_size, patch_size, None)

print("Setting up Network")
pixels_with_road = load.load_road_pixel_index(REAL_PATH)
generator = proc.generate_patch(x_img,y_img,patch_size,batch_size=batch_size,augment=False, defer=True)
# generator = proc.generate_patch_road(x_img,y_img, pixels_with_road,patch_size,batch_size=batch_size,augment=False, defer=True)

model.fit_generator(generator,steps_per_epoch=16,epochs=500000,callbacks=[checkpointer,plotter],use_multiprocessing=True)
