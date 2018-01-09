import caffe
import numpy as np
import yaml
import time

class PyPixelShuffleLayer(caffe.Layer):

    def __pixShuf(self, I):
        
        r = self.scale_factor
        n, c, h, w = I.shape
        oc = c / r**2

        I = I.reshape((n, oc, r**2, h, w)) # 1x2x4x540x960
        I = I.transpose(0,1,3,4,2)
        I = I.reshape((n, oc, h, w, r, r))  # 1x2x540x960x2x2
        I = I.transpose(0,1,2,4,3,5)
        I = I.reshape((n, oc, h*r, w*r))
        
        return I
        
 
    def __pixUnShuf(self, I):
        r = self.scale_factor
        n, oc, h, w = I.shape
        c = oc * r**2
        h /= r
        w /= r        
        I = I.reshape((n, oc, h, r, w, r))
        I = I.transpose(0,1,3,5,2,4)
        I = I.reshape((n, c, h, w))

        return I

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("PyShuffle Expects 1 input")
        if len(top) != 1:
            raise Exception("Expects 1 output")

        layer_params = yaml.load(self.param_str)
        self.scale_factor = layer_params['scale_factor']

    def forward(self, bottom, top):
        top[0].data[...] = self.__pixShuf(bottom[0].data)
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.__pixUnShuf(top[0].diff)

    def reshape(self, bottom, top):
        self.n, self.c, self.h, self.w = bottom[0].shape
        assert (self.c % (self.scale_factor ** 2)) == 0
        self.out_channels = self.c / (self.scale_factor ** 2)        
        self.out_h = self.h * self.scale_factor
        self.out_w = self.w * self.scale_factor
        top[0].reshape(self.n, self.out_channels, self.out_h, self.out_w)


