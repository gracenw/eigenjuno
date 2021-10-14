# Object Detection and Classification for JunoCam Images
- Python notebook containing current research for using eigenimages to detect features on Jupiter's surface
- Now includes stitching workload (as of 7/26/21) based on [this](https://www.reddit.com/r/space/comments/ewl69t/my_frustrating_walkthrough_to_processing_junocams/) reddit post
- Code based on tutorials found [here](https://pythonmachinelearning.pro/face-recognition-with-eigenfaces/) and [here](https://www.betterdatascience.com/eigenfaces%E2%80%8A-%E2%80%8Aface-classification-in-python/), and algorithm found [here](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

## Previous Milestones
- Eigenimaging proves to be easily trainable using SVM on JunoCam images
  - Most effective classifier is C-Support Vector Classifier
- Contrasted images shown to be more accurate than non-contrasted
- Algorithm can detect and classify individual cropped images containing white storms with 90% accuracy
- Chopping up new images and running through divide-and-conquer pipeline
- Divide-and-conquer pipeline detects features within an image with moderate accuracy
- Test divide-and-conquer pipeline on more new images

## Current Work
- Add more no storm images to dataset
- Implement divide-and-conquer at multiple resolutions

## Future Work
- Connect features to their actual coordinates with SPICE data
- Produce tracking results across perijoves / introduce some sort of memory capability
- Add more features to dataset
- Include ability to add eigenfaces to eigenspace for newly discovered features
