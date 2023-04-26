# DL-skin-lesions

## Overview

### Team:
Grzegorz Maliniak, Piotr Słowik
### Description:
Skin lesion recognition using CNN 
### Dataset:
HAM10000 - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## How to use

1. Download the repository by cloning it or downloading the ZIP file and extract it to a folder of your choice. Open terminal in repos folder.

2. Create a virtual environment and activate it. This can be done by running the following commands in the terminal:
```
python -m venv env
source env/bin/activate
```

3. Install the project dependencies by running the following command in the terminal:
```
pip install -r src/requirements.txt
```

4. Download the dataset into the `data` directory by running the following command in the terminal:
```
dvc pull
```

5. Preprocess the images and create a Torch DataLoader for the dataset by running the following command in the terminal:
```
kedro run --pipeline='data_loader'
```
This command will also split the data into training and testing datasets and create training weights.

6. Create the CNN model by running the following command in the terminal:

```
kedro run --pipeline='model_create'
```

7. To view the training metrics and visualizations you need to be logged in WandB. To log in to WandB, run the following command in the terminal: 

```
wandb login
```

8. Train the CNN model. All the metrics will be logged to a WandB project, and the trained model will be saved inside Kedro.

```
kedro run --pipeline='model_train'
```

9. Once training is complete, you can view the training metrics and visualizations by logging in to the WandB project.

10. To create documentation, use the following command in the terminal:
```
kedro build-docs
```


### Short version

1. Download the repository by cloning it or downloading the ZIP file and extract it to a folder of your choice. Open terminal in repos folder.

2. Type in terminal:

```
python -m venv env
source env/bin/activate
pip install -r src/requirements.txt
dvc pull
wandb login
kedro run
```
