from config.config_node import ConfigNode

config = ConfigNode()
config.mode = 'Classification'

config.model = ConfigNode()
config.model.model_test_dir = 'models/final.pth'
config.model.use_tensorboard = False
config.model.num_classes = 2
config.model.class_names = ['Forest', 'River']
config.model.image_size = 64
config.model.device =  "cpu"
config.model.save_model_dir = "models/"

config.dataset = ConfigNode()
config.dataset.train_dataset_path = 'C:\\Users\\s84171040\\Documents\\RetiSpec\\RGBIRDataset\\train\\'
config.dataset.test_dataset_path = 'C:\\Users\\s84171040\\Documents\\RetiSpec\\RGBIRDataset\\val\\'
config.dataset.val_ratio = 0.2

config.hyperparameters = ConfigNode()
config.hyperparameters.epochs = 51
config.hyperparameters.batch_size = 8
config.hyperparameters.shuffle = True
config.hyperparameters.num_workers = 0
config.hyperparameters.pin_memory = False
config.hyperparameters.drop_last = False
config.hyperparameters.learning_rate = 0.001
config.hyperparameters.optimizer = "Adam"
config.hyperparameters.momentum = 0.9
config.hyperparameters.adam_beta1 = 0.9
config.hyperparameters.adam_beta2 = 0.999
config.hyperparameters.gamma = 0.1
config.hyperparameters.weight_decay = 0.00005
config.hyperparameters.milestones = [5,20,50]
config.hyperparameters.scheduler = "STEP"
config.hyperparameters.val_freq = 5
config.hyperparameters.save_freq = 5
def get_default_config():
    return config.clone()



