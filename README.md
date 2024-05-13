This repository is part of my Final Year Project, titled: "Deep Learning-Based Image Segmentation of Membranous Urethra in MR Imaging"

It contains the following folders:

- `/data` - contains the slice images and binary segmentation masks in different formats (both lines, right lines, different thicknesses, closed area), as well as Patient ID and MUL value information in `info.csv`
- `/data_processing` - data preprocessing for initial DICOM images to NumPy
- `/model` - ML pipeline for model training and evaluation, for both segmentation and regression
