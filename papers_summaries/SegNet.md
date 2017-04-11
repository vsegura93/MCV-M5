#### SegNet: deep encoder-decoder architecture for multi-class pixelwise segmentation 
SegNet architecture was designed as a semantic pixel-wise segmentation oriented deep fully convolutional neural network. It 
was developed as an encoder net followed by its decoder and a pixel-wise classiﬁcation layer on top of that. Its major 
contribution lies the decoder’s upsampling method for the lower resolution input feature maps. The process is non-linear making
use of pooling indices computed in the max-pooling step of the corresponding encoder. By doing so, the result is a smaller 
network in terms of number of trainable parameters compared to other architectures being trainable end-to-end using stochastic 
gradient descent.

The network was developed focusing on its application in road scene understanding. The need to map low resolution features to 
input resolution was the main goal. With this, they could perform a pixel-wise classiﬁcation good enough to produce accurate 
boundaries. The idea was inspired by unsupervised feature learning architectures. 

Its key learning module encoder-decoder’s first part consists of blocks applying convolution with a ﬁlter bank, element-wise tanh non-linearity, max-pooling and 
sub-sampling to obtain the feature maps. Particularly, the indices of the max locations computed during pooling are stored and 
passed to the decoder. This is done with each sample and allows the decoder to upsample the feature maps using these indices. 
By storing only the indices of the maximum feature values at each step before max-pooling, only 2 bits for every 2x2 pooling 
window is spent in memory being much more efficient. The cons about this solution is a slight loss of accuracy though. When 
the decoder takes the input, it upsamples following the corresponding indices, creating a sparse feature map. These are then 
convolved with a trainable decoder ﬁlter bank to produce dense feature maps and then a batch normalization is applied. The
high dimensional feature representation at the output of the ﬁnal decoder is fed to a trainable soft-max classiﬁer.

In pursuit of a decoder method comparison as even as possible, the developers confront a smaller version of SegNet against a 
custom small version of FCN. This way, they can expose the differences between their approach, and the ones that use decoding 
methods along the lines of FCN. The results achieved show the important reduction in memory consumption due to the system 
adopted to index the feature maps responses and use the indices afterwards for upsampling. The use of a small version of the 
architecture is also applied in the early stages of training, seeking a converging configuration. The dataset used is CamVid 
and also SUN RGB-D Indoor, performing pre-processing on the inputs to improve performance. One of said measures is local
contrast normalization to the RGB input, batch image shuffling and even applying median frequency balancing for the loss 
function weights. 

Their results prove SegNet to achieve competitive results agains CRF based networks. SegNet can absorb a 
large training set and generalize well to unseen images lessening the contribution of the prior (CRF). Nevertheless, 
networks which store the totality of the feature maps are still higher in performance. The strongest point about SegNet is 
its lower memory consumption and relatively faster training.
