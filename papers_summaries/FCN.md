#### Fully convolutional networks for semantic segmentation (Long et al. CVPR, 2015)

This works that directly explores the convolutional architectures shows how just by using a set of basic components as: 
convolution, pooling and activation function, we are able to give an output to a given input image, that has the same
dimensions. Each layer of data in the structure have 3 dimensions of size h*w*d, where 'h' and 'w' are the spatial dimensions 
and 'd' the channel dimension. In the case of the first layer, 'h' and 'w' corresponds to the pixels in the image and 'd' the 
number of color channels. In the next layers, as the architecture is based on convolutions (and the result of the convolution 
is a feature map with small dimensions in 'h' and 'w' than the input image), 'h' and 'w' become smaller in each layer and 'd' 
corresponds to the number of kernels in each layer (each kernel, convolutional matrix give a 2D feature map). The result of 
pass the image through the network is a coarse feature map with a receptive field higher than one pixel (as the convolutional 
kernels used are 3 by 3 minimum).

This implies to have as a result a coarse and small feature map that can be reescaled in order to have the same dimensions as 
the input, but obtaining but results when labeling each pixel. 

By this reason a new architecture was implemented. In this case, a set of skip connections was added in order to improve the 
results. The most important think is that deep layers give us coarse feature maps, but that has learn semantic information 
about the image (are able to assign a class to a pixel), while the shallower (first) layers give fine and appearance 
information of the image, as it receptive field is smaller. By combining both information the output is finer and the network 
is able to assign a label to each pixel obtaining a good performance.

To do that the authors pick architecture networks designed for the classification task (the one that give a single label to 
the whole image) and modify the architecture. The main problem of the networks for classification purpose is that have at the 
end one or many fully connected layers to predict a fixed number of outputs, having just one dimension. By this way we loss
spatial information of the image. The authors of that work proposed to delete fully connected layers and add a 1 by 1 
convolutional layer to have a fixed number of channel outputs that corresponds to the classes to predict (per each pixel). 
That give a coarse and small sized feature map. By adding the skip connection (that adds to the output a sampled version of a 
previous convolutional size we can have semantic and appearance information.

Once we have finer feature map, they needed to resize the image in order to have an image with the same dimensions as the 
input image. This is possible by applying an interpolation of the finer feature map, but this can be translated and expressed 
as an operation known as deconvolution. This is a convolution applied with an upsampled version of the output. 

Applying that authors tried with VGG-16, AlexNet and GoogLeNet architectures, performing the correspondent changes explained 
before.

To train that remodeled network their propose to fine-tune the model updating the weights of all layers, because that complex 
models requires a lot of time and samples to be trained. By loading the pre-trained weights of that networks trained with 
ImageNet, we can have good results with less examples and time. Added layers cannot be initialized with pre-trained weights 
because are not available but different ways to initializate them can be applied.
