import numpy as np
import os
import handlers.args as args
import handlers.load as load
from handlers.proc import MyProcess as proc

arg = args.get_args()

#----Load Data---
print("Loading Images")
rgb = load.load_data(arg.input_folder+"/rgb")
rgb_mean = load.get_mean(arg.input_folder+"/rgb")

x = rgb
arg.mean = rgb_mean

del rgb
y = load.load_data(arg.input_folder+"/y")

#validation
rgb_ng = load.load_data(arg.input_folder+"/rgb_ng",end=1)
x_ng = rgb_ng

del rgb_ng
y_ng = load.load_data(arg.input_folder+"/y_ng")

proc.setup(x,y, x_ng, y_ng, arg.patch_size,arg.batch_size, arg.mean, y_is_flat=True, one_hot = False)

del y, y_ng

#----KERAS ENV-------
os.environ["CUDA_VISIBLE_DEVICES"]=str(arg.gpu)

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model

import keras.backend as K
def uoi_by_mse(true, pred):
    btrue = K.cast(true, bool)
    btrue = K.cast(btrue, float)
    bpred = K.cast(pred, bool)
    bpred = K.cast(bpred, float)
    intersection = btrue * bpred
    notTrue = 1 - btrue
    union = btrue +(notTrue * bpred)
    return K.sum(union) / K.sum(intersection) * K.mean(K.square(pred - true), axis=-1)
    # return (1 + K.epsilon() - K.sum(intersection)/(K.sum(union)+K.epsilon())) * K.mean(K.square(pred - true), axis = -1)

#----COMPILE MODEL---
if arg.cont:
    print("Loading Model")
    model = load_model(arg.output_folder+'/model.hdf5')
else:
    #----MODEL MODULE---
    from models import loader
    module = loader.load(arg.model)
    print("Generating Model")
    model = module.build(proc.input_shape,len(proc.y_classes))
    print("Compiling Model")
    model.compile('RMSprop', uoi_by_mse)


#----CHECKPOINT CALLBACKS -----
checkpointer = ModelCheckpoint(filepath=arg.output_folder+'/model.hdf5', verbose=0, save_best_only=False)
#PLOT
from handlers.plots import PlotLoss as plot
plotter = plot(arg, proc)

#_!_!_!_!_!_!_RUN NETWORK_!_!_!_!_!_
print("Running Network")
model.fit_generator(proc.generate_patch(augment=arg.augment),steps_per_epoch=64,epochs=500000,callbacks=[checkpointer,plotter],use_multiprocessing=False)
