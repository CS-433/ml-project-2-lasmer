 # Road Segmentation Project

This project focuses on the segmentation of roads in images using three different models.
 The following is an organizational structure and explanation of each file within the project.

## File Organization

- `Networks/`: Directory containing the different neural network models that were used for road segmentation (Unet/ LinkNet/  GCDCNN)
- `test_set_images/`: Contains the test images that are input to the models to evaluate their segmentation capabilities.
- `test_set_masks/`: Includes the ground truth masks generated for the test images (for evaluation purposes) .
- `Loader.py`: Python script responsible for loading and preprocessing the data before it is fed into the models.
- `augment_deterministic.py`: Script that applies deterministic augmentations to the images for consistent model training.
- `augment_random.py`: Applies random augmentations to the images to increase the robustness of the models during training.
- `helpers.py`: Contains utility functions and helpers used across different scripts in the project.
- `mask_to_submission.py`: Converts segmentation masks output by the models into a submission format for evaluation on AICrowd.
- `requirements.txt`: Lists all Python dependencies required for the project.
- `submission_to_mask.py`: Converts submission files back into segmentation masks for analysis.
- `train.py`: The main script to train the models on the dataset.
- `train_fold.py`: Script used for training the models with cross-validation folds.
- `README.md`: Markdown file providing an overview and documentation for the project.


## Data
The original data can be downloaded from AI Crowd using this link : https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files
It consists of 100 images of shape 400x400 and their groundtruths. 
1) Download and unzip training.zip to access the training dataset and train the road segmentation models.
2) After training, run the models on the test set provided in test_set_images.zip to generate segmentation masks.
3) Use `mask_to_submission.py` to convert the masks to the submission format, which would be structured similarly to `sample_submission.csv`.
4) (Optional) For additional analysis or verification, convert submissions back into masks using `submission_to_mask.py`.

### Data Augmentation 
We provide two augmentation files since the original data is not enough for training. 
You can use them by running (one of the following or both)
-  `augment_deterministic.py`
-  `augment_random.py` 

This will create a new folder:


|--data_augmented  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--training  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--augmented_ground_truth  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--augmented_images  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--validation  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--augmented_ground_truth  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--augmented_images  


For additional data, you can change NB_AUGMENTATIONS in `augment_random.py` (by default=3)

## Addtionnal Data:
You can download other datasets we used through this link : https://drive.google.com/drive/folders/1gbACOtl24JjgL8i28iU3mKlfwh4-ppGK?usp=share_link

## Usage
To use this project, first clone the repository, make sure you have python 3.9 or above installed on your system and install all dependencies listed in `requirements.txt` by running:

```bash 
git clone https://github.com/CS-433/ml-project-2-lasmer.git
```
```bash
cd ml-project-2-lasmer
```
```bash
pip install -r requirements.txt
```

### Training Models
Once everything is set up, you can run the training scripts (`train.py` or `train_fold.py`) to train the models with your dataset.
### Example Commands for Model Training using Unet:

```bash
python train.py --model UNet --batch_size 8 --epochs 50 --loss dice --lr 3e-4
```

```bash
python train_fold.py --k_folds 4 --model UNet --batch_size 8 --epochs 50 --loss dice --lr 3e-4
```

Note: <br>
- For regular `train.py`: you should have the following structure: 

|--data  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--training  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--labels  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--images  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--validation  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--labels  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--images  



- For k-fold `train_fold.py`: you do not need a validation folder.

### Predictions
We created a file `predict.py` to compute the final masks.
When running the script, you can specify which models to use for making predictions. For instance:

- To make predictions using only the UNet model (the default setting):
     ```bash
  python predict.py
     ```
- To use both UNet and GCDCNN:
  ```bash
   python predict.py --use_unet True --use_GCDCNN True
  ```
- To use all three models:
   ```bash
  python predict.py --use_unet True --use_GCDCNN True --use_linknet True
   ```
- To use UNet with cropping and TTA:
  ```bash
   python predict.py --use_crop True --use_TTA True
  ```

Prior to making predictions with a particular model, ensure that you have successfully trained the model and saved it in the .pt format. The training and saving process is automated and can be performed using the `train.py` or `train_fold.py` scripts.
