# Object Detection and Classification for JunoCam Images
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
eigejuno  
    |____ eigenimages.ipynb  
    |____ utils.py  
    |____ DATA  
    |      |____ RAW  
    |      |      |_________ PERIJOVE-XX  
    |      |________ TEST  
    |      |________ TRAIN  
    |                  |____ ONE  
    |                  |____ ZERO  
    |                  |____ PROCESS  
    |____ FIGURES  
    |____ LOGS  
    |____ MODELS  
    |____ STITCHING  
              |____ main.ipynb  
              |____ stitch.py  
              |____ KERNELS  
                       |____ CURRENT  

