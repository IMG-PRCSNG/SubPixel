import numpy as np
import sys
import os

def runNet(input_blob, net, output_blob):
    
    net.blobs['data'].data[...] = input_blob

    output = net.forward()  # run once before timing to set up memory

    return output[output_blob][:]

if __name__ == "__main__":

    model_def = sys.argv[1]
    caffe_root = './' 
    if os.path.isdir(caffe_root):
        sys.path.insert(0, caffe_root + 'python')
        try:
            import caffe
            caffe.set_mode_cpu()


        except ImportError:
            raise ImportError(caffe_root + " doesn't appear to be the caffe root")
        
    net = caffe.Net(model_def, caffe.TRAIN)

    """
    # Test it with (in **ipython**)
    run path/to/pythonScript path/to/prototxt

    a = np.arange(1*18*30*30).reshape((1,18,30,30))
    s = runNet(a, net, 'shuf')

    # s should have a shape of (1,2,90,90)
    # Backward pass of s should give a
    u = net.backward(**{net.outputs[0]: s})

    # u will have a shape of (1, 18, 30, 30)

    """


