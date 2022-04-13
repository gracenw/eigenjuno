# Object Detection and Classification of JunoCam Images
- Python notebook containing current research for using eigenimages to detect features on Jupiter's surface
- Stitching workload adapted from [this repository](https://github.com/cosmas-heiss/JunoCamRawImageProcessing/)
- Code based on algorithm found in [this paper](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

## Abstract
The JunoCam is a moderate-resolution camera mounted on the Juno spacecraft, launched by the National Aeronautic and Space Administration (NASA) in 2011, used to generate images of a previously unseen quality for both research and public outreach. Detailed Red, Green, and Blue (RGB) images from JunoCam can be used in conjunction with object detection algorithms to track visual features of Jupiter’s atmosphere. The positioning and movement of these features, such as circumpolar cyclones, hazes, and more, can provide insight into the nature of Jupiter’s atmosphere. This positional data can be used in conjunction with infrared and microwave data from Jovian InfraRed Auroral Mapper (JIRAM) and Microwave Radiometer (MWR) to give a more comprehensive assessment of Jupiter’s atmospheric qualities. The object detection method to be used in this research is eigenimaging, an approach adapted from the facial recognition method of eigenfaces. This methodology uses the eigenvectors of images projected onto a known eigenspace to match images from a new perijove capture to those in a dataset of previous perijoves. The efficient detection and classification of features across perijove captures is the ultimate goal of this method, wherein further analysis will extract the desired positional data. The objective of this proposed research is to prove that eigenimaging is a viable tracking method for white storms of various sizes on Jupiter’s surface and to provide a framework for expanding the algorithms tracking mechanisms and trackable features. 

<!-- 
## Previous Milestones
- Eigenimaging proves to be easily trainable using SVM on JunoCam images
- Contrasted images shown to be more accurate than non-contrasted
- Algorithm can detect and classify individual cropped images containing white storms with 90% accuracy
- Chopping up new images and running through divide-and-conquer pipeline
- Divide-and-conquer pipeline detects features within an image with moderate accuracy
- Test divide-and-conquer pipeline on more new images
- Add more no storm images to dataset
- Implement divide-and-conquer at multiple resolutions
- Connect features to their actual coordinates with SPICE data
- Include ability to add eigenfaces to eigenspace for newly discovered features
## Future Work
- Produce tracking results across perijoves / introduce some sort of memory capability
- Add more features to dataset 
-->

## File Structure
eigenjuno  
&nbsp;&nbsp;&nbsp;&nbsp;|____ main.ipynb -> example code using eigenjuno module for training/testing  
&nbsp;&nbsp;&nbsp;&nbsp;|____ eigenjuno.py -> module containing all needed functions and packages for detection/classification  
&nbsp;&nbsp;&nbsp;&nbsp;|____ FIGURES  
&nbsp;&nbsp;&nbsp;&nbsp;|____ LOGS  
&nbsp;&nbsp;&nbsp;&nbsp;|____ MODELS  
&nbsp;&nbsp;&nbsp;&nbsp;|____ DATA  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ TEST  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ RAW  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ PERIJOVE-#  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ TRAIN  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ ONE  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ ZERO  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ PROCESS  
&nbsp;&nbsp;&nbsp;&nbsp;|____ STITCHING  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ main.ipynb -> example code for generating full color images, adapted from [this repository](https://github.com/cosmas-heiss/JunoCamRawImageProcessing/)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ stitch.py -> module for stitching images, all credit to [this repository](https://github.com/cosmas-heiss/JunoCamRawImageProcessing/)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|____ KERNELS  

## Required Packages
After installing Anaconda for Linux, all required packages can be downloaded with:  
```bash
conda install --file package-list.txt
```

## Models & Training Data
Zip files containing all training data, separated into ONE and ZERO folders, can be found at [this Google Drive link](https://drive.google.com/drive/folders/1YbvWGE1yyTrjiDjXWEzdFfO4TXsCOeQ7?usp=sharing).  
The optimal PCA and SVM models can be found under MODELS and will load automatically using the given test function.

## Generating Images
To use the stitching workload, follow the instructions below:  
1. Download metadata and images from [the official Mission Juno website](https://www.missionjuno.swri.edu/junocam/processing?source=junocam&ob_from=&ob_to=&perpage=100) - select specific perijoves via the menu on the left, under 'Mission Phases', and be sure to take note of the date the image was captured  
2. Using the file structure above, drop the ####-Data.zip and ####-ImageSet.zip files in the appropriate perijove directory  
3. Keeping in mind the date the image was captured, find the [CK files](https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/ck/) the encompass the capture date in the range of dates in the file names
4. The same process is used to the [SPK files](https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/spk/)
5. Drop both of these kernels in STICHING/KERNELS
6. Run the first 4 cells in the STITCHING/main.ipynb notebook, making sure to change the 'perijove' variable in the second cell to the appropriate number
7. Check to make sure the appropriate files were unzipped and placed in the perijove directory - ####-Metadata.json and a PNG with a naming structure like 'JNCE_2019149_20C00030_V01-raw.png'
8. Run the remaining cells in the stitching notebook; to maximize efficiency, have multiple raw images and their metadata counterparts reading to process, as these can be done simultaneously via multithreading

## Using the eigenjuno Module
Examples are included in main.ipynb for using the functions provided in the eigenjuno module, namely for training, testing, and data visualization. Each cell in the notebook will explain what is happening - just be sure to follow to file structure above, and download the required models and training set.
