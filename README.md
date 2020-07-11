# Detection and Analysis of Pitting Corrosion using RCNN
A research on Detection and analysis of pitting corrosion in metals exposed to alkaline medium of varying concentration using a Fast RCNN model in collaboration with CSIR - Central Electro Chemical Research Institute.

<p align="middle">
    <img src="https://user-images.githubusercontent.com/18646185/87218756-4553da00-c373-11ea-827f-ed6c835a5fa9.png" width="300" />
</p>

### Run the project
##### Prerequisites
- python 3.5 or above
- Tensorflow

##### Train
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_coco.config
```

##### Evaluate
```
python eval.py --logtostderr --pipeline_config_path=training/ssd_mobilenet_v1_coco.config --checkpoint_dir=training/ --eval_dir=eval/
```

### Abstract
Pitting corrosion is a form of corrosion where metal is selectively removed at small areas in a corrosive environment. Most commonly used metals like Steel, Aluminium are highly susceptible to pitting corrosion whose structural integrity is monumental in various fields of engineering. With the help of Object detection algorithm using Convolutional Neural Networks, we have done image analysis of Scanning Electron Microscope images of stainless-steel specimens to detect and count pits of various sizes and shapes to help study the characteristics of pitting corrosion. The neural network approach is visually compared with the conventional Morphology image processing technique which is found to be less accurate in the detection of corrosion pits. 

### Introduction
The World Corrosion Organization estimates the worldwide price of corrosion to be US $2.5 trillion annually, which an oversized portion of this - the maximum amount as twenty five percent - might be eliminated by applying easy, well-understood hindrance techniques. Corrosion hindrance mustn't be thought of entirely a monetary issue, but also one of health and safety. The advancements of technology in especially in the field of artificial intelligence and machine learning can be used to take preventive measures in a timely manner to reduce monetary and health damages. Pitting corrosion is one of the most common forms of corrosion in metals like Steel and Aluminium. A small pit in these metals can cause failure of an entire engineering system/structure. We are using a Region based Convolutional Neural Network (RCNN) to detect the corroded pits in steel specimens exposed to Chloride medium. We train this variant of the neural network with the SEM dataset of the specimens. The pits are manually labelled using bounding boxes in the image and trained. The trained network is tested using test samples and the network draws a bounding box around every detected pit. The results are compared with the groundtruths of the test images and the accuracy of the trained model is calculated. 

### Dataset

The dataset is comprised of SEM (Scanning Electron Microscope) images of metallic plates exposed to alkaline medium of varying concentrations for varying durations. The dimension of the SEM image is 512 x 512 (pixels). The SEM images are further divided into segments of 16 x 16 images.

<p align="middle" float="left">
    <img src="https://user-images.githubusercontent.com/18646185/87218049-fa36c880-c36c-11ea-85b9-4d29ee3df66d.jpg" width="200" />
    <img src="https://user-images.githubusercontent.com/18646185/87218094-4aae2600-c36d-11ea-9583-f459a039b353.jpg" width="200" /> 
</p>

##### Sample Preparation
304 type stainless steel specimens having nominal composition of 18% Cr; 8% Ni; 2% Mn; 0.10%N; 0.03 % S; 0.08% C; 0.75 % Si; 0.045 % P.  Specimens of dimension (1 x 1 cm<sup>2</sup> ) were used as a working area for SEM observation. The surface preparation of the mechanically abraded specimens was carried out using different grades of silicon carbide emery paper (up to 1200 grit) and subsequent cleaning with acetone and rinsing with double-distilled water were done before each experiment.
In this study, at the end of every 5 minutes potentiostatic polarization of samples was taken at 300mv and 400mv Vs Saturated Calomel Electrode (SCE) were used to investigate the pitting corrosion of 304 stainless steel in 3.5 wt% NaCl solution and 0.1N Ferric chloride solution. The surface morphology of pitted surface was imaged at 30x magnification using Tescan Vega 3 SBH Scanning Electron Microscopy (SEM).


##### Annotations
The corroded pits were annotated using [labelImg](https://github.com/tzutalin/labelImg)
<p align="middle">
<img src="https://user-images.githubusercontent.com/18646185/87218215-4d5d4b00-c36e-11ea-8977-15878a775f15.png" width="600"/>
</p>

### Morphological image processing approach
Before we trained and tested the neural network, we used a morphological image processing approach for pit detection. Morphological operations apply a structuring component to an input image, making an output image of identical size. During a morphological operation, the value of pixel in the output image relies on a comparison of the corresponding pixel in the input image with its neighbors. We used Watershed algorithm for this study. Watershed is a transformation on grayscale images. The aim of this method is to section the image, generally when 2 regions-of-interest are near to each other — i.e., their edges touch. This technique of transformation treats the image as a topographical map, with the intensity of every pixel representing the height. for example, dark areas can be intuitively thought of to be ‘lower’ in height and can represent troughs. On the other hand, bright areas may be thought of to be ‘higher’, acting as hills or as a mountain ridge. Assume that a supply of water is placed within the structure basins (the areas with low intensity). These basins are flooded and areas wherever the floodwater from completely different basins meet are known. Barriers in the form of pixels are built in these areas. Consequently, these barriers act as partitions within the image, and therefore the image is segmented.

As you can see from the below images, this method resulted in a very inaccurate segmentation of pits.

<p align="middle">
    <span>Normal SEM of a stainless steel plate</span>
    <br/>
    <br/>
    <img src="https://user-images.githubusercontent.com/18646185/87218367-982b9280-c36f-11ea-9110-310f23798691.png" width="200" />
    <br/>
    <br/>
    <span>After thresholding and removing noise</span>
    <br/>
    <br/>
    <img src="https://user-images.githubusercontent.com/18646185/87218450-6bc44600-c370-11ea-9459-a1d2092e1a78.png" width="400" />
</p>

### RCNN model
<p align="middle">
    <img src="https://user-images.githubusercontent.com/18646185/87218532-4552da80-c371-11ea-887b-b44b2b29e73b.png"/>
</p>

##### Validation
Testing of the trained model is done with the help of test samples with groundtruth boxes for evaluation purposes. The evaluation metrics are obtained once the testing is complete. The model was tested with a sample of 40 images with 766 pits. The output of the RCNN used in this study is the list bounding boxes which will ideally contain all the corrosion pits in an image. Denoting boxes as pit or non-pit
can yield three potential results, with the latter two being sources of error: true positive (TP)—correctly classifying a region as a pit; false positive (FP)—incorrectly classifying a background region as a pit as well as multiple detection of the same pit; and false negative (FN)—incorrectly classifying a spike as a background region. In contrast, true negative (TN)—correct classification of background is always ’zero’ and is not required in this binary classification problem where foreground is always determined for object detection. In order to quantify our errors, the validation metrics are based on the concepts
of precision, recall, accuracy and the F1 score, which are defined as follows:

Precision = TP/(TP + FP)  measures how many of the detected regions are actually pits.

Recall = TP/(TP + FN) measures how many of the pits the image are detected.

Accuracy = (TP + TN)/(TP + TN + FP + F) implies the model’s performance.

F1 score = 2(Precision x  Recall)/(Precision + Recall)  is the harmonic mean of Precision and Recall. It is a useful measure to observe the mode’s robustness.

### Results

<p align="middle">
    <span>Evaluated sample image with detection boxes</span>
    <br/>
    <br/>
    <img src="https://user-images.githubusercontent.com/18646185/87218756-4553da00-c373-11ea-827f-ed6c835a5fa9.png" width="200" />
    <br/>
    <br/>
    <span>Evaluated sample image with groundtruth boxes and detected boxes.</span>
    <br/>
    <br/>
    <img src="https://user-images.githubusercontent.com/18646185/87218755-4422ad00-c373-11ea-959e-a323baaaa794.png" width="200" />
    <br/>
    <br/>
    <span>Predicted pits vs Actual pits</span>
    <br/>
    <br/>
    <img src="https://user-images.githubusercontent.com/18646185/87218844-1db14180-c374-11ea-8c5c-ee6ce3c7ab99.jpg" width="600" />
</p>
