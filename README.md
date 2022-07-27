# RetiSpec-Project

** About the Project ****

This project contains code to train and test a CNN based neural network for binary classification. 

* Code Structure
*   The following describes the folder structure and associated files

* config: 
  *  This project uses YACS configuration. Yacs allows keeping tracking of changes in hyperparameters making it easier for experimentation.
  *  A default configuration can be found in config/defaults.py file. Any changes with respect to the default file can be mentioned in the .yaml files.
  *  In this project, train.yaml contains parameters that are specific to only training and test.yaml contains parameters specific for testing. 
  *  The parameters defined in the .yaml files overwrite the default parameters (defined in defaults.py file) in the train.py and test.py script. 

* model_training:
  * This folder consists of all the training related scripts including network architecture and the dataloader. 
  * dataset_loader.py, dataset_creater.py, one_class_dataset.py and transforms.py are scripts associated with the data loading 
  * network.py contains the code for the CNN

* train.py: Responsible for training the CNN network
* test.py: Responsible for testing the CNN network by loading the trained model

* Additional Folders
  * models:
    * This folder consists of the models that are saved during training and the final trained model
  * logs:
    * This folder consists of tensorboard file that can be loaded to view the train/val behavior

* Network Architecture
  * Input to the network is RGB and IR image. Output is probability of whether the RGB and IR images belong to Class Forest (0) or River (1). 
  * The network defined in network.py consists of three main modules: two feature extractors and one projection module. 
  * The projection module is followed by an output layer. 
  * Feature Extractor
    * The input to each of the two feature extractors is an image of shape 64x64xC and output is a feature map with dimensions 8*8*32
    * Value of C depends on the input type.
    * For RGB image, C is 3 while for NIR image, C is 1.
    * Each of the Feature extractor consists of 3 CNN layers (in total 6 CNN layers in the network). 
    * CNN layer is followed by BatchNorm layer, Relu Activation function and MaxPooling Layer. 
    * The number of channels increases with each of the CNN layers.
  * The output of the two feature extractors which are 2D feature maps are flattened resulting in a 1d vector each of size 512
  * The two 1d vectors of size 512 are fused together by elementwise multiplication. This ensures that the network only uses the important features from the two input images for learning to differentiate between the two classes.
    
  * Projection Module consists of 2 FC layers.
    * Input to the projection module is a 1D vector of size 512 and output is a 1D vector of size 32. 
    * Each of the FC layers consists a linear layer followed by Batchnorma and Relu Activation function. 
  
* Data Augmentation During Training using albumentations library:
  * Avoided the use of color based augmentations since not sure the effect on the RGB and IR image
  * Applied Blurring, Distortion and Spatial Transform which include scaling.shifting and rotation. 

* Dataset Splitting
  * Dataset provided consists of train and val folders. Each of the two folders consists of equal number of images for each of the two classes
  * Since no test dataset is provided, the val images were used for testing only
  * The train images were split into train and validation with the ratio of split being 80:20

To look at the train/val accuracy and loss curves, please use tensorboard. You will find the tensorboard file in the logs folder
Trained the network for 20 epochs only. Could have trained the network longer since validation loss is still reducing. Usually validation loss is used as the criteria for stopping the training process

* Performance Evaluation
  *  Since dataset is balanced, accuracy can be used as a criteria for evaluation
  *  For a more detailed analysis, confusion matrix is computed
  *  From the confusion matrix below, one can see that there is only 1 image where misclassification is reported. 

![Alt text](confusion_matrix.png?raw=true "Confusion Matrix")



