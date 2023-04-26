import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule
from torchmetrics import Accuracy

class LesionDetector(LightningModule):
    """
    Three-layered Convolutional Neural Network made for
    skin lesion type classification.
    """
    def __init__(self, image_size, channels_out, kernel_size, fc_features, num_classes, learning_rate, training_weights):

        self.learning_rate = learning_rate
        super().__init__()
        multiclass_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.metrics = [multiclass_accuracy]

        self.class_weights = training_weights

        conv1_out_shape = (channels_out, (image_size[0]-kernel_size+1)//2, (image_size[1]-kernel_size+1)//2)
        conv2_out_flat_shape = (channels_out*2 * ((((image_size[0]-kernel_size+1)//2)-kernel_size+1)//2) * ((((image_size[0]-kernel_size+1)//2)-kernel_size+1)//2))

        self.ln1 = nn.LayerNorm((3,) + tuple(image_size))
        self.conv1 = nn.Conv2d(3, channels_out, kernel_size=kernel_size)
        self.dropout1 = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ln2 = nn.LayerNorm(conv1_out_shape)
        self.conv2 = nn.Conv2d(channels_out, channels_out*2, kernel_size=kernel_size)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.ln3 = nn.LayerNorm((conv2_out_flat_shape))
        self.fc1 = nn.Linear(conv2_out_flat_shape, fc_features)
        self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_features, num_classes)

        self.save_hyperparameters()

    def log_metrics(self, x, y, loss, prefix):
        """
        Returns nothing, logs metrics.
        """
        for metric in self.metrics:
            metric.to(self.device)
            self.log(prefix + metric._get_name(), metric(x, y), on_step = False, on_epoch = True)
        self.log(prefix + 'Loss', loss, on_step = False, on_epoch = True)


    def training_step(self, batch, batch_idx):
        """
        Returns value of loss function for training batch given in arguments.
        """
        x, y = batch
        x = self.forward(x)
        loss = nn.CrossEntropyLoss(weight = self.training_weights)(x,y)
        
        self.log_metrics(x, y, loss, 'Train ')

        return loss
    
    def on_train_epoch_start(self):
        """
        Returns nothing, initializes training weights of classification labels.
        """
        self.training_weights = torch.tensor(self.class_weights, device = self.device)
        
    
    def validation_step(self, batch, batch_idx):
        """
        Returns value of loss function for validation batch given in arguments.
        """
        x, y = batch
        x = self.forward(x)
        loss = nn.CrossEntropyLoss()(x,y)
        
        self.log_metrics(x, y, loss, 'Validation ')

        return loss
    
    def forward(self, x):
        """
        Returns output produced by model for input 'x' given in arguments.
        The architecture of model are two simple convolutional layers with 
        dropout and pooling. last layer is a simple Linear with dropout.
        """
        x = self.ln1(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.ln2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = self.ln3(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


    def configure_optimizers(self):
        """
        Returns configured optimizers for the model.
        """
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def create_model(model_params, loader_params, train_weights):
    """
    Returns new LesionDetector model.
    """
    create_params = model_params['create_params']

    return LesionDetector(loader_params['image_size'], create_params['channels_out'], create_params['kernel_size'], create_params['fc_features']
                    , loader_params['num_classes'], create_params['learning_rate'], train_weights)



