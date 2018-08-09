# Project : Self-Driving Car Assignment
In This project, I have implemented some simplified component of a self_driving car. 

## Dependencies
This project requires Python 3.5 and Python libraries spicified in requirements.txt 

##1. White line extraction:
I have used only six images to reduce time of tratement and tests 

the different steps in this algorithm that will enable us to identify and classify lane lines look as follows:
    * Convert original image to HSL
    * Isolate yellow and white from HSL image
    *Convert image to grayscale for easier manipulation
    - Apply Gaussian Blur to smoothen edges
    - Apply Canny Edge Detection on smoothed gray image
    - Trace Region Of Interest and discard all other lines identified by our previous step that are outside this region
    - Perform a Hough Transform to find lanes within our region of interest and trace them in red
    - Separate left and right lanes
    - Extrapolate them to create two smooth lines

for testing ,I have used some images I have taken them from udacity  datasets. 
to start the test: 
```
    cd tests    
    python3 line_extraction_test.py 
```


##2. traffic sign 

    - Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each signs ..
    - Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the German Traffic Sign Dataset.: 
    - Converting to grayscale 
    - Normalizing the data to the range (-1,1)
    - Four functions for augmenting the dataset: random_translate, random_scale, random_warp, and random_brightness

 [Download the datase](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip)
 create traffic_sign_datasets file in Project and extract data in this file  ( you will need those file in line 166 and 167 of traffic_sign.py so give the right path..)
to start the test: 
```
    cd self_driving_car_modules     
    python3 trafffic_sign.py 
```

##3. Car detection using machine learning and object detection algorithm 
this algorithm of vehicle detector is developped  by employing a conventional computer vision technique called Histogram of Oriented Gradients (HOG), combined with a machine learning algorithm called Support Vector Machines (SVM).
In order to test this algorithm, I have used different Datasets
-[GTI Vehicle Image Database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) : download this datasets and exract it Project/project_datasets file 
- [the project video form kitti](http://www.cvlibs.net/datasets/kitti/) 

The alogrithm is based: 
- Dataset exploration
- Feature Extraction
- apply Histogram of Oriented Gradients and Color bins toinput image to create features
- Exploration Of Features
- explore the result of our HOG operations.[explained here](https://www.learnopencv.com/histogram-of-oriented-gradients/)
- Finding Suitable Color Space
- Classification
- Heatmap Thresholding¶

to start the test: 
```
    cd self_driving_car_modules     
    python3 object_detection.py 
```



### thankk you 




