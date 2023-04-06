"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.6
"""


from pathlib import Path
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch import loggers as pl_loggers

def train_model(model, train_params, train_dataset, test_dataset):

    train_data_loader = DataLoader(train_dataset, batch_size = train_params['batch_size'], num_workers=12)
    test_data_loader = DataLoader(test_dataset, batch_size = train_params['batch_size'], num_workers=12)
    csv_logger = pl_loggers.CSVLogger(save_dir = Path.cwd() / "logs/")

    print(f'EPOCHS: {train_params["epochs"]}')

    t = pl.Trainer(max_epochs=train_params['epochs'], logger=csv_logger)
    t.fit(model=model, train_dataloaders=train_data_loader, val_dataloaders=test_data_loader)
    
    return model