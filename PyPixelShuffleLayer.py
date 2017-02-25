import caffe
import numpy as np
import yaml

class PyPixelShuffleLayer(caffe.Layer):

    def __pixShuf(self, I):
        #(bs, c, a, b)
        r = self.scale_factor
        n, c, h, w = I.shape
        oc = c / r**2

        I = np.split(I, oc, axis=1)
        print len(I), " x ", I[0].shape
        I = np.concatenate([np.expand_dims(i, axis=1) for i in I], axis=1)

        print I.shape
        I = I.transpose(0,1,3,4,2)
        I = I.reshape((n, oc, h, w, r, r))
        I = np.split(I, h, axis=2)
        I = np.concatenate([np.squeeze(i, axis=2) for i in I], axis=3)
        I = np.split(I, w, axis=2)
        I = np.concatenate([np.squeeze(i, axis=2) for i in I], axis=3) 
        return I

    def __pixUnShuf(self, I):
        #(bs, oc, ar, br)
        r = self.scale_factor
        n, oc, h, w = I.shape
        c = oc * r**2
        h /= r
        w /= r
        
        I = np.split(I, w, axis=3)
        I = np.concatenate([np.expand_dims(i, axis=2) for i in I], axis=2)
        I = np.split(I, h, axis=3)
        I = np.concatenate([np.expand_dims(i, axis=2) for i in I], axis=2)
        I = I.reshape((n, oc, h, w, r**2))
        I = I.transpose(0,1,4,2,3)
        I = np.split(I, oc, axis=1)
        I = np.concatenate([np.squeeze(i, axis=1) for i in I], axis=1)
        return I

        


    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("PyShuffle Expects 1 input")
        if len(top) != 1:
            raise Exception("Expects 1 output")

        layer_params = yaml.load(self.param_str)
        self.scale_factor = layer_params['scale_factor']

        self.n, self.c, self.h, self.w = bottom[0].shape

        assert (self.c % (self.scale_factor ** 2)) == 0

        self.out_channels = self.c / (self.scale_factor ** 2)        
        self.out_h = self.h * self.scale_factor
        self.out_w = self.w * self.scale_factor


    def forward(self, bottom, top):
        top[0].data[...] = self.__pixShuf(bottom[0].data)
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = self.__pixUnShuf(top[0].diff)
        pass

    def reshape(self, bottom, top):
        top[0].reshape(self.n, self.out_channels, self.out_h, self.out_w)


# For debug reasons
#def apixShuf(I, r ):
#    assert I.shape[1] % r**2 == 0
#    n, c, h, w = I.shape
#    oc = c / r**2
#
#    I = np.split(I, oc, axis=1)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    # oc, (bs, r**2, a, b)
#
#    I = np.concatenate([np.expand_dims(i, axis=1) for i in I], axis=1)
#    print "Shape after concat : ", I.shape
#    # (bs, oc, r**2, a, b)
#
#    I = I.transpose(0,1,3,4,2) 
#    # Transpose b01c (bs, oc, a, b, r**2)
#    print "Shape after transpose : ", I.shape
#
#    I = I.reshape((n, oc, h, w, r, r))
#    print "Shape after reshape : ", I.shape
#    # (bs, oc, a, b, r, r)
#    
#    I = np.split(I, h, axis=2)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    # a, (bs, oc, 1, b, r, r)
#     
#    I = np.concatenate([np.squeeze(i, axis=2) for i in I], axis=3)
#    print "Shape after concat : ", I.shape
#    # (bs, oc, b, ar, r)
#        
#    I = np.split(I, w, axis=2)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    # b, (bs, oc, 1, ar, r)
#    
#    I = np.concatenate([np.squeeze(i, axis=2) for i in I], axis=3) 
#    print "Shape after concat : ", I.shape
#    #(bs, oc, ar, br)
#    return I
#
#    #return np.expand_dims(I, axis=1) # (bs, 1, ar, br)
#
#    #return I.reshape((self.n, self.h*r, self.w*r, 1)).transpose(0,3,1,2) 
#
#def apixUnShuf(I, r):
#    assert I.shape[2] % r == 0
#    assert I.shape[3] % r == 0
#    #(bs, oc, ar, br)
#    n, oc, h, w = I.shape
#    c = oc * r**2
#    h /= r
#    w /= r
#    
#    I = np.split(I, w, axis=3)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    #b, (bs, oc, ar, r)
#
#    I = np.concatenate([np.expand_dims(i, axis=2) for i in I], axis=2)
#    print "Shape after concat : ", I.shape
#    #(bs, oc, b, ar, r)
#
#    I = np.split(I, h, axis=3)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    #a, (bs, oc, b, r, r)
#
#    I = np.concatenate([np.expand_dims(i, axis=2) for i in I], axis=2)
#    print "Shape after concat : ", I.shape
#    #(bs, oc, a, b, r, r)
#
#    I = I.reshape((n, oc, h, w, r**2))
#    print "Shape after reshape : ", I.shape
#    #(bs, oc, a, b, r**2)
#
#    I = I.transpose(0,1,4,2,3)
#    print "Shape after transpose : ", I.shape
#    #(bs, oc, r**2, a, b)
#
#    I = np.split(I, oc, axis=1)
#    print "Shape after split : ", len(I), " x ", I[0].shape
#    #oc, (bs, 1, r**2, a, b)
#    
#    I = np.concatenate([np.squeeze(i, axis=1) for i in I], axis=1)
#    print "Shape after concat : ", I.shape
#    #(bs, c, a, b)
#
#    return I
#

