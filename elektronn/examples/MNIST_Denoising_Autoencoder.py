# -*- coding: utf-8 -*-
"""
ELEKTRONN - Neural Network Toolkit

Copyright (c) 2014 - now
Max-Planck-Institute for Medical Research, Heidelberg, Germany
Authors: Marius Killinger, Gregor Urban
"""

import matplotlib.pyplot as plt

from elektronn.training.traindata import MNISTData
from elektronn.net.convnet import MixedConvNN
from elektronn.net.introspection import embedMatricesInGray

# Load Data #
data = MNISTData(path=None,
                 convert2image=False,
                 shift_augment=False)

# Create Autoencoder #
batch_size = 100
cnn = MixedConvNN((28**2), input_depth=None)
cnn.addPerceptronLayer(n_outputs=300, activation_func="tanh")
cnn.addPerceptronLayer(n_outputs=200, activation_func="tanh")
cnn.addPerceptronLayer(n_outputs=50, activation_func="tanh")
cnn.addTiedAutoencoderChain(n_layers=None,
                            activation_func="tanh",
                            input_noise=0.3)
cnn.compileOutputFunctions(target="regression")  #compiles the cnn.get_error function as well
cnn.setOptimizerParams(SGD={'LR': 5e-1, 'momentum': 0.9}, weight_decay=0)

print "training..."
for i in range(40000):
    d, l = data.getbatch(batch_size)
    loss, loss_instance, time_per_step = cnn.trainingStep(d, d, mode="SGD")

    if i % 100 == 0:
        print "update:", i, "; Training error:", loss

cnn.layers[3].input_noise.set_value(0.0)
loss, _, test_predictions = cnn.get_error(data.valid_d.reshape((10000, 784)), data.valid_d.reshape((10000, 784)))
print "Final error:", loss
print "Done."

plt.figure(figsize=(14, 6))
plt.subplot(121)
images = embedMatricesInGray(data.valid_d[:200].reshape((200, 28, 28)), 1)
plt.imshow(images, interpolation='none', cmap='gray')
plt.title('Data')

plt.subplot(122)
recon = embedMatricesInGray(test_predictions[:200].reshape((200, 28, 28)), 1)
plt.imshow(recon, interpolation='none', cmap='gray')
plt.title('Reconstruction')

d_v = data.valid_d
pred = test_predictions.clip(d_v.min(), d_v.max())
recon = embedMatricesInGray(pred[:200].reshape((200, 28, 28)), 1)
plt.imshow(recon, interpolation='none', cmap='gray')

# Use pre-trained weights from AE to train a classifier
cnn.saveParameters('AE-pretraining.param', layers=cnn.layers[0:3])  # only save the weights from the compression

cnn2 = MixedConvNN((28**2), input_depth=None)
cnn2.addPerceptronLayer(n_outputs=300, activation_func="tanh")
cnn2.addPerceptronLayer(n_outputs=200, activation_func="tanh")
cnn2.addPerceptronLayer(n_outputs=50, activation_func="tanh")
cnn2.addPerceptronLayer(n_outputs=10, activation_func="tanh")
cnn2.compileOutputFunctions(target="nll"
                            )  #compiles the cnn.get_error function as well
cnn2.setOptimizerParams(SGD={'LR': 0.005, 'momentum': 0.9}, weight_decay=0)
cnn2.loadParameters('AE-pretraining.param')
print "training..."
for i in range(10000):
    d, l = data.getbatch(batch_size)
    loss, loss_instance, time_per_step = cnn2.trainingStep(d, l, mode="SGD")
    error = cnn2.get_error(d, l)[1]

    if i % 100 == 0:
        print "update:", i, "; Training error:", error, "NLL:", loss

print "Final error:", loss
print "Done."
