# Re_X - Delineation of Road Networks Using Deep Residual Neural Networks and Iterative Hough Transform

## Folder structure

```bash
.
├── post-processing & evaluation
│   ├── *.cpp
│   └── *.h
├── README.md
├── training & testing
│   ├── models
│   │   ├── static
│   │   │   └── *.py
│   │   └── *.py
│   ├── handlers
│   │   └── *.py
│   ├── train.py
│   ├── patch_test.py
│   └── README.md
├── data
│   ├── rgb_ng		# Imagery (RGB) for testing
│   │   └── *.png/*.tif/*.jpg
│   ├── out
│   │   ├── post_processing_result
│   │   │   └── *.png/*.tif/*.jpg
│   │   ├── graph
│   │   │   ├── pred	# .graph files for predictions
│   │   │   │   └── *.graph
│   │   │   └── truth	# .graph files for ground truth
│   │   │       └── *.graph
│   │   ├── mask
│   │   │   ├── pred 	# Masks of predictions
│   │   │   │   └── *.png/*.tif/*.jpg
│   │   │   └── truth	# Masks of ground truth
│   │   │       └── *.png/*.tif/*.jpg
│   │   └── errorImg	# Difference maps
│   │       ├── *.png/*.tif/*.jpg
│   │       └── rgb 	# Difference maps on RGB imagery
│   │           └── *.png/*.tif/*.jpg
│   ├── rgb 		# Imagery (RGB) for training
│   │   └── *.png/*.tif/*.jpg
│   ├── all_patches	# All patches for testing and post-processing
│   │   ├── pred
│   │   │   └── *.png/*.tif/*.jpg
│   │   └── rgb
│   │       └── *.png/*.tif/*.jpg
│   ├── y_ng		# Ground truth (grayscale) for testing images
│   │   └── *.png/*.tif/*.jpg
│   ├── y 			# Ground truth (grayscale) for training
│   │   └── *.png/*.tif/*.jpg
```
