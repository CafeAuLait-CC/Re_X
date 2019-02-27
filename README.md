# Re_X  
Delineation of Road Networks Using Deep Residual Neural Networks and Iterative Hough Transform

## Train the model
1. Go to `training & testing` folder and run `python3 mean.py -i ../data/rgb/`
2. Run `python3 train.py -n TRAINING_NAME`, the trained model will be saved in `results/TRAINING_NAME/model.hdf5`

## Inference
1. Create folder `data/rgb_ng/patches_to_predict/`
2. Go to `post-processing & evaluation` folder, in `main.cpp` file run the `generateAllPatches()` function to get all patches for inference. After this, you will get a bunch of 200x200 image patches saved in `data/rgb_ng/patches_to_predict/`, the file names of these images represent their location in the original image tile.
3. Go to `training & testing` folder and run `python3 patch_test.py -n TRAINING_NAME` to inference the road map using the `TRAINING_NAME` model, segmentation result will be saved in `results/TRAINING_NAME/result_on_patches/`

## Post-processing
1. Create folder `results/TRAINING_NAME/post_processing_result/`
2. Go to `post-processing & evaluation` folder, in `main.cpp` file run the `cleanUpHoughLineImage()` function to get vectorized result images (final results). Output images will be saved in the folder `post_processing_result/`.

## Evaluation
1. Create folders `results/TRAINING_NAME/post_processing_result/errorImg/`
2. Go to `post-processing & evaluation` folder, in `main.cpp` file run the `startEval()` function, a evaluation table called `eval.txt` will be saved in `errorImg` folder
3. In `main.cpp` file run the `drawDiffMapOnRGB()` function, the difference image will be drew on the rgb imagery and saved in the `errorImg` folder
