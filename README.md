# Object Detection and Classification for JunoCam Images
- Python notebook containing current research for using eigenimages to detect features on Jupiter's surface
- Does not include stitching workload
- Code based on tutorials found [here](https://pythonmachinelearning.pro/face-recognition-with-eigenfaces/) and [here](https://www.betterdatascience.com/eigenfaces%E2%80%8A-%E2%80%8Aface-classification-in-python/), and algorithm found [here](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

## Previous Milestones
- Eigenimaging proves to be easily trainable using SVM on JunoCam images
  - Most effective classifier is C-Support Vector Classifier
- Contrasted images shown to be more accurate than non-contrasted
- Algorithm can detect and classify individual cropped images containing white storms with 90% accuracy
- Chopping up new images and running through divide-and-conquer pipeline

## Current Work
- Divide-and-conquer pipeline is still producing strange results (there's an error somewhere in classification)

## Future Work
- Test divide-and-conquer pipeline on more new images
- Add more features to dataset
- Include ability to add eigenfaces to eigenspace for newly discovered features
- Connect features to their actual coordinates with SPICE data
- Produce tracking results across perijoves / introduce some sort of memory capability
