import mynet
import resnet50
import unet
import classnet

def gen(input_shape,model,nclasses=177,p_out=None):
    if model=='mynet':
        if not nclasses:
            return mynet.build_rmse(input_shape)
        else:
            return mynet.build(input_shape,nclasses)
    if model=='mynet_card':
        return mynet_card.build(input_shape,nclasses)    
    if model=='resnet50':
        return resnet50.build(input_shape,nclasses)
    if model=='unet':
        return unet.build(input_shape,nclasses)
    if model=='classnet':
        return classnet.build(input_shape,nclasses)
    if model=='carnet':
        return carnet.build(input_shape,nclasses)
