import caffe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

caffe.set_mode_cpu()

network_path = "nets/my_train_val_deploy.prototxt"
weights_path = "nets/my_train_val_iter_500.caffemodel"
net = caffe.Net(network_path, weights_path, caffe.TRAIN)    # load to training phase. Use .TEST for testing. Unsure what
                                                            # this means in this case. However with the regularNOT deploy version,
                                                            # it changes thigs, see the .prototxt

# you can also load a net without specifying the weights file, but I am not sure what it does: probably initialises randomly.
# loaded_network = caffe.Net(network_path, caffe.TEST) # caffe.TEST for testing

# configure preprocessing
imName = 'jpgs/cfgRank3400.jpg'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('lmdb/meanConfig.npy').mean(1).mean(1))
#transformer.set_transpose('data', (2, 0, 1)) # only needed for color images
#transformer.set_channel_swap('data', (2, 1, 0)) # only needed for color images
transformer.set_raw_scale('data', 255.0)
net.blobs['data'].reshape(1, 1, 60, 100)

# load the image in the data layer
im = caffe.io.load_image(imName)
net.blobs['data'].data[...] = transformer.preprocess('data', im)


# compute output
output = net.forward()
output_prob = output['softOut']
print 'predicted probabilities are:', output_prob

# plot the input and a neuron of the first convolution layer, and one in conv layer 4
plt.figure(1)
plt.imshow(net.blobs['data'].data[0,1])
plt.figure(2)
plt.imshow(net.blobs['conv1'].data[0,1])
plt.figure(3)
plt.imshow(net.blobs['conv4'].data[0,1])
plt.show()

