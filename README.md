# Mars Surface and Curiosity Image Classification
### Using a convolutional neural network

by Zachary Pruessner, for Springboard Final submission

**Dataset DOI**

10.5281/zenodo.1049137

**Scientific Paper with additional details on the data** 

Kiri L. Wagstaff, You Lu, Alice Stanboli, Kevin Grimes, Thamme Gowda,
and Jordan Padams. "Deep Mars: CNN Classification of Mars Imagery for
the PDS Imaging Atlas." Proceedings of the Thirtieth Annual Conference
on Innovative Applications of Artificial Intelligence, 2018.

## Objective

The objective of this project is to build a convolutional neural network capable of classifying images from mars.

## Data Wrangling

Initally, the data was setup for another computer system and needed to have the pathing corrected. This was fairly simple and only required a single helper function.

From there, the three seperate csv files were merged and saved for further processing.

## Exploratory Data Analysis

The analysis of this data was quite simple, as the data was fairly straight forward and organized. Although, it was found that the dataset was slightly imbalanced.
This came in the form of numerous images of the ground. Significantly more than any other category. Another issue that will be addressed in the next section is that of
image size. They are all around 256x256, but about half of them are not. Therefore it will be necessary to transform the images to a consistent size of 256x256.

## Pre-Processing

For pre-processing, the images were resized to 256x256 and normalized. This way, all of the values are between 1 and 0. And all of the images are of the same size.

## Modeling

The model consists of the following architecture, which is very similar to the AlexNet.
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 62, 62, 96)        34944     
                                                                 
 batch_normalization (BatchN  (None, 62, 62, 96)       384       
 ormalization)                                                   
                                                                 
 max_pooling2d (MaxPooling2D  (None, 30, 30, 96)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 30, 30, 256)       614656    
                                                                 
 batch_normalization_1 (Batc  (None, 30, 30, 256)      1024      
 hNormalization)                                                 
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 14, 14, 256)      0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 384)       885120    
                                                                 
 batch_normalization_2 (Batc  (None, 14, 14, 384)      1536      
 hNormalization)                                                 
                                                                 
 conv2d_3 (Conv2D)           (None, 14, 14, 384)       1327488   
                                                                 
 batch_normalization_3 (Batc  (None, 14, 14, 384)      1536      
 hNormalization)                                                 
                                                                 
 conv2d_4 (Conv2D)           (None, 14, 14, 256)       884992    
                                                                 
 batch_normalization_4 (Batc  (None, 14, 14, 256)      1024      
 hNormalization)                                                 
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 6, 6, 256)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 9216)              0         
                                                                 
 dense (Dense)               (None, 4096)              37752832  
                                                                 
 dropout (Dropout)           (None, 4096)              0         
                                                                 
 dense_1 (Dense)             (None, 4096)              16781312  
                                                                 
 dropout_1 (Dropout)         (None, 4096)              0         
                                                                 
 dense_2 (Dense)             (None, 25)                102425    
                                                                 
=================================================================
Total params: 58,389,273
Trainable params: 58,386,521
Non-trainable params: 2,752
_________________________________________________________________
```

For its loss function it uses Categorical Crossentropy. The optimizer is 'adam', it comes standard from the TensorFlow Libray.

When training the model, 50 epochs were used, with a batch size of 64, and a validation split of 10%.


## Performance

The trained model sits at loss: 0.2115, accuracy: 0.9784, val_loss: 0.7868, val_accuracy: 0.9519

These are great numbers to start out with. Unfortunately, the model performance is somewhat unknown. The prediction on the test data returned 100% accuracy.
Which is hard to believe and requires further investigation. 

## Ideas for Further Research

Further steps might include the building and training of an unsupervised model, and the deployment of the supervised model. Additional research might take
the form of different model structures and their respective performance.