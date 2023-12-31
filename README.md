# About
This repository is a fine-tuned version of FocusOnDepth (FOD) model in [here](https://github.com/antocad/FocusOnDepth).
It can estimate the depth of a RGB facial image.

The training data is an augmented version of BU3DFE dataset.
We render the RGB and depth images at 7 views per 3D face model.

# Inference
Users should fill the background with black and align the input image with the given landmarks.
The output image has 4 channels: depth, x, y, z, where (x, y, z) are components of normal vector.

# TODOs
- [ ] Release the checkpoint
- [ ] Delete the unused codes
- [ ] Release the landmarks for aligning
