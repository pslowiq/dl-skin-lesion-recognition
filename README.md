# DL-skin-lesions

## Overview

### Team:
Grzegorz Maliniak, Piotr Słowik
### Description:
Skin lesion recognition using CNN 
### Dataset:
HAM10000 - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

## How to use

### Download data, run the project
Download repository, create venv, install dependencies from requirements.txt and then type in terminal:
```
dvc pull
kedro run
```

### Run specific pipelines
#### Convert images to numpy and make train/test datasets
```
kedro run --pipeline='data_loader'
```

#### Create simple model - CNN with 1-dense layer
```
kedro run --pipeline='model_create'
```

#### Train model
```
kedro run --pipeline='model_train'
```
