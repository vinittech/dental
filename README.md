# Root Classification Task #


* [**Top level Directory**](#top-level-directory)
* [**Installation**](#installation)
* [**Inference**](#inference)
* [**Training**](#training)

## Top level Directory

    .
    ├── checkpoint              # Contains training checkpoint
    ├── Data                    # Training Data (contains 3 Folders and Orientation.txt File)
    ├── Test                    # Some Test Data from Google
    ├── Results                 # Screenshots of the training (Includes both VGG & CustomCNN results)
    ├── dataloader.py           # Custom Data loading process for training
    ├── inference.py            # Running an Inference on an Image
    ├── model.py                # Network Architecture
    ├── environment.yml         # Contains all the required packages
    ├── train.py                # Training setup
    └── README.md


## Installation

1. Download the repository<br>

2. Create a conda environement<br>
  `conda env create -n 3d -f environment.yml`

3. To activate the environment:<br>
  `conda activate 3d`


## Inference

* The script `inference.py` processes an image and returns the prediction.
```
arguments:
  --help              # show this help message and exit
  --imagePath         # Path to the Image
  --rootDir           # Directory path to the training Data (To extract Idx_to_classes) mapping
  --checkpoint        # Checkpoint of the model
```

Example usage (Default arguments already provided in the script)         

```
python3 inference.py
```

## Training

* Create a directory named `Data` that contains the **3** categories(cat,horse,squirrel) and the `orientations.txt` file of the data.

        Data                              
            ├── multiple   
            ├── one     

            
* Run the script `train.py` script.

```
arguments:
  --help               # show this help message and exit
  --batchSize          # Training Batch Size
  --totalEpochs        # Number of epochs to train for
  --snapshots          # Snapshot Frequency of the checkpoint
  --lr                 # Learning Rate
  --threads            # Number of threads for data loader to use
  --rootDir            # Directory path to the training Data
  --rotationTxt        #  Txt file on rotation of images
  --saveCheckpoint     #  Directory to store checkpoint

```

Example usage (Default arguments already provided in the script)           

```
python3 train.py  
```
