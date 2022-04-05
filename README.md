# Object Detection and Classification for JunoCam Images
- Python notebook containing current research for using eigenimages to detect features on Jupiter's surface
- Stitching workload adapted from [this repository](https://github.com/cosmas-heiss/JunoCamRawImageProcessing/)
- Code based on algorithm found in [this paper](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf)

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
