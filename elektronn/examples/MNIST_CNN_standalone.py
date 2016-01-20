# -*- coding: utf-8 -*-
"""
ELEKTRONN - Neural Network Toolkit
Copyright (c) 2015 Gregor Urban, Marius Killinger
"""

from elektronn.training.traindata import MNISTData
from elektronn.net.convnet import MixedConvNN

# Load Data #
data = MNISTData(path=None,
                 convert2image=True,
                 shift_augment=False)

# Create CNN #
batch_size = 100
cnn = MixedConvNN((28, 28), input_depth=1)
cnn.addConvLayer(10,
                 5,
                 pool_shape=2,
                 activation_func="abs")  # (nof, filtersize)
cnn.addConvLayer(8, 5, pool_shape=2, activation_func="abs")
cnn.addPerceptronLayer(100, activation_func="abs")
cnn.addPerceptronLayer(80, activation_func="abs")
cnn.addPerceptronLayer(
    10,
    activation_func="abs"
)  # need 10 outputs as there are 10 classes in the data set
cnn.compileOutputFunctions()
cnn.setOptimizerParams(SGD={'LR': 1e-2, 'momentum': 0.9}, weight_decay=0)

print "training..."
for i in range(5000):
    d, l = data.getbatch(batch_size)
    loss, loss_instance, time_per_step = cnn.trainingStep(d, l, mode="SGD")

    if i % 100 == 0:
        valid_loss, valid_error, valid_predictions = cnn.get_error(data.valid_d, data.valid_l)
        print "update:", i, "; Validation loss:", valid_loss, "Validation error:", valid_error * 100., "%"

loss, error, test_predictions = cnn.get_error(data.test_d, data.test_l)
print "Test loss:", loss, "Test error:", error * 100., "%"
print "Done."
