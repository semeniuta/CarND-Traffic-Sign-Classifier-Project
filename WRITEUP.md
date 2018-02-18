# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[TrainImages]: ./demo-images/train_images.png "Training images"

[NormTrainImages]: ./demo-images/normalized_images.png "Normalized training images (red channel)"

[WebImagesPredict]: ./demo-images/web_images_prediction.png "Web images and their predictions"

[WebImagesProbab]: ./demo-images/web_images_probab.png "Softmax probabilities of web images' predictions"

---

### Dataset exploration

The loaded data from the German traffic signs dataset is summarized as follows:

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

The distribution of classes in the training data set is summarized in the following table:

```
                                                  SignName  NumExamples
ClassId
0                                     Speed limit (20km/h)          180
1                                     Speed limit (30km/h)         1980
2                                     Speed limit (50km/h)         2010
3                                     Speed limit (60km/h)         1260
4                                     Speed limit (70km/h)         1770
5                                     Speed limit (80km/h)         1650
6                              End of speed limit (80km/h)          360
7                                    Speed limit (100km/h)         1290
8                                    Speed limit (120km/h)         1260
9                                               No passing         1320
10            No passing for vehicles over 3.5 metric tons         1800
11                   Right-of-way at the next intersection         1170
12                                           Priority road         1890
13                                                   Yield         1920
14                                                    Stop          690
15                                             No vehicles          540
16                Vehicles over 3.5 metric tons prohibited          360
17                                                No entry          990
18                                         General caution         1080
19                             Dangerous curve to the left          180
20                            Dangerous curve to the right          300
21                                            Double curve          270
22                                              Bumpy road          330
23                                           Slippery road          450
24                               Road narrows on the right          240
25                                               Road work         1350
26                                         Traffic signals          540
27                                             Pedestrians          210
28                                       Children crossing          480
29                                       Bicycles crossing          240
30                                      Beware of ice/snow          390
31                                   Wild animals crossing          690
32                     End of all speed and passing limits          210
33                                        Turn right ahead          599
34                                         Turn left ahead          360
35                                              Ahead only         1080
36                                    Go straight or right          330
37                                     Go straight or left          180
38                                              Keep right         1860
39                                               Keep left          270
40                                    Roundabout mandatory          300
41                                       End of no passing          210
42       End of no passing by vehicles over 3.5 metric ...          210
```

To get a glimpse of the appearance of training images, every first image for each class is visualized:

![alt text][TrainImages]

### Data normalization

The data normalization strategy used in this project is based on computing the mean intensity per each channel in RGB (implemented in `tfnet.get_image_data_mean_per_channel`), which is then subtracted from the original per-channel pixel values. The resulting difference is further divided by 255. Implementation of this normalization procedure is found in `tfnet.image_data_scaling`.

The resulting normalized red channel for the images is visualized below:

![alt text][NormTrainImages]

### Model Architecture

The initial prototyping was done on the LeNet implementation from the corresponding lab, with feeding the traffic sign data (see `LeNet-Lab-TrafficSingsData.ipynb`). Further, the same functionality was decomposed into a collection of helper functions in `tfnet.py` to facilitate greater configurability and code separation.

Following the LeNet architecture, the convolutional network used in this project is comprised of a sequence of convolutional layers (including 2D-convolution, ReLU-activation and pooling, see `tfnet.create_conv2d_layer`), followed by the flattening-operator and a sequence of fully-connected layers (with ReLU-activation except the final layer, see `tfnet.create_fully_connected_layer`).

Each convolutional layer is configured by three integers describing its output tensor shape. Likewise, a fully-connected layer is configured by an integer value corresponding to the length of the output 1D tensor. By considering the input layer's shape to be always equal to the image dimension (32, 32, 3), the rest of the layers can be succinctly specified by a dictionary as follows (the configuration below is the final convnet architecture used in this project):

```
{
    'conv_layers': [[28, 28, 6], [10, 10, 16]],
    'fc_layers': [120, 84, 43]
}
```

The `tfnet.create_convnet` function creates a TensorFlow network with the structure described above, and is parametrized by the latter form of dictionary.

In the training phase, outputs of every convolutional layer is fed to a dropout node, with every dropout node configured (via the `keep_prob` parameter) by a common `tf.float32` placeholder. Every weights tensor is randomly initialized by sampling from `tf.truncated_normal` with the common `mean` and `stddev` parameters.

### Grid search experiments

To find out how different hyperparameters affect the convnet performance, two grid search experiments were performed (the first on a local computer, and the second on an AWS GPU instance). Both experiments ran for 20 epochs and kept track of how classification accuracy evolved throughout each epoch.

The first experiment (`grid-search-1.py`) varied different learning rates, batch sizes and dropout probabilities:

```python
rates = (0.001, 0.01, 0.1)
batch_sizes = (64, 128, 256)
dropout_probs = (0.4, 0.5, 0.6)
```

As one can see in `grid-search-1/analyze_grid_search_1.ipynb` notebook, the highest accuracy (0.957) was achieved with batch size of 64, dropout probability of 0.5, and learning rate of 0.001. In general it was observed that the small learning rate of 0.001 and dropout probability of either 0.5 or 0.6 yield better prediction.

In the second experiment (`grid-search-2.py`), different convnet configurations were compared, given the learning rate of 0.001, dropout probability of 0.5 and batch size of 128:

```python
nn_configurations = (
    {'conv_layers': [(28, 28, 6), (10, 10, 16)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
    {'conv_layers': [(28, 28, 7), (10, 10, 17)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
    {'conv_layers': [(28, 28, 8), (10, 10, 18)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
    {'conv_layers': [(28, 28, 9), (10, 10, 19)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1},
    {'conv_layers': [(28, 28, 10), (10, 10, 20)], 'fc_layers': [120, 84, 43], 'mean': 0., 'stddev': 0.1}
)
```

All options led to similar classification accuracies.

Accuracy of the chosen model with regards to training, validation and testing sets is summarized below:

```
Training set accuracy: 0.990
Validadition set accuracy: 0.956
Test set accuracy: 0.933
```

### Testing the model on new images

To test how the trained model generalizes to new data, a set of publicly available (Creative Commons-licensed) images of German traffic signs was found on the Web. The image regions with the signs of interest were cropped, resized to (32, 32, 3), and saved as the new test images.

The trained convnet predicted 7/12 images correctly. The original set of the web images is show below (an image is marked with OK if predicted correctly, and with the incorrect prediction otherwise):

![alt text][WebImagesPredict]

The top 5 softmax probabilities for each web image are show in the following figure (green bar charts correspond to correctly classified images and red bar charts otherwise):

![alt text][WebImagesProbab]

As can be seen from the last figure, in general, a correct prediction is characterized by more "certain" distribution of softmax probabilities, while failed prediction has less certainty. An odd exception is the wrongly classified stop sing. One possibility for this fail might be unbalance in the number of training images (690 for "Stop" and 1890 for "Priority road"). Similar explanation might hold true also for wrong classification of "Children crossing" (480 training examples). The proportion of the area of the actual sign region seems to have a big influence on accuracy as well.
