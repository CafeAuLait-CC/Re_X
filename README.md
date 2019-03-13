# [Re_X](http://theictlab.org/lp/2019Re_X/)  
Delineation of Road Networks Using Deep Residual Neural Networks and Iterative Hough Transform  
## Usage
### Main Dependencies
1. Keras + TensorFlow  
2. NumPy, Matplotlib, Scikit-image
2. OpenCV for C++ & Python3  
3. CMake & make  

### Dataset preparation
1. Create a folder `data/` under this project directory
2. Create 4 folders under `data/`: `rgb`, `y`, `rgb_ng` and `y_ng`, put RGB imagery into `rgb` folder, ground truth image into `y` folder. `rgb_ng` and `y_ng` are for test set. You can use .jpg, .png or .tif format. 

### Train the model
1. Go to `training & testing` folder and run `python mean.py -i ../data/rgb/`
2. Run `python3 train.py -n TRAINING_NAME`, the trained model will be saved in `results/TRAINING_NAME/model.hdf5`

	```
	$ cd training\ \&\ testing/
	$ python mean.py -i ../data/rgb/
	$ python train.py -n TRAINING_NAME
	```

### Inference
1. Go to `post-processing & evaluation` folder, compile and run the `main.cpp` file to get all patches for inference. 

	```
	$ cd Re_X/post-processing\ \&\ evaluation/
	$ mkdir build && cd build
	$ cmake ..
	$ make && cd ..
	$ ./Re_X 0 TRAINING_NAME
	```
	After this, you will get a bunch of 200x200 image patches saved in `Re_X/data/rgb_ng/patches_to_predict/`, the file names of these images represent their location in the original image tile.
2. Go to `training & testing` folder and run the test program to inference the road map using the `TRAINING_NAME` model, segmentation result will be saved in `Re_X/results/TRAINING_NAME/result_on_patches/`  

	```
	$ cd training\ \&\ testing/
	$ python patch_test.py -n TRAINING_NAME
	```

### Post-processing
1. Go to `post-processing & evaluation` folder,  run the `./Re_X` program in post-processing mode to get vectorized result images (final results). Output images will be saved in the folder `Re_X/results/TRAINING_NAME/post_processing_result/`.

	```
	$ cd Re_X/post-processing\ \&\ evaluation/
	$ ./Re_X 1 TRAINING_NAME
	```

### Evaluation
1. Go to `post-processing & evaluation` folder, run the `./Re_X` program in evaluation mode, a evaluation table called `eval.txt` will be saved in `Re_X/results/TRAINING_NAME/post_processing_result/errorImg/`, and the difference image will be drew on the rgb imagery and saved in the same folder

	```
	$ cd Re_X/post-processing\ \&\ evaluation/
	$ ./Re_X 2 TRAINING_NAME
	```
	
### Help
Usage for the C++ program:

```
$ ./Re_X 

Usage: ./Re_X mode -n model_name [...opts]

    mode:  	0: prepare the inference data		# generateAllPatches()
	   	1: post-processing & refinement		# cleanUpHoughLineImage()
	   	2: evaluation				# startEval() & drawDiffMapOnRGB()

    -n:  the folder name used to save the trained model.

    opts:  -w --image_width		(default 8192)
	   -h --image_height		(default 8192)
	   -c --patch_cols		(default 81 -- file name from 0 to 80)
	   -r --patch_rows		(default 81 -- file name from 0 to 80)
	   -i --input_folder		(leave empty to use default setting)
	   -o --output_folder		(leave empty to use default setting)
```

#### IMPORTANT: To use this software, YOU MUST CITE the following in any resulting publication:  
```
@article{xu2019,
title={Delineation of Road Networks Using Deep Residual Neural Networks and Iterative Hough Transform},
author={Xu, Pinjing and Poullis, Charalambos},
year={2019}
}
```
