import yaml
import os
import argparse




# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--layers",
                    type=int,
                    default=1,
                    help="Number of hidden layers.")
parser.add_argument("--neurons",
                    type=int,
                    default=1,
                    help="Number of neurons on each hidden layer.")
parser.add_argument("--scale",
                    type=int,
                    default=1,
                    help="Scale of the joints limits.")

args = parser.parse_args()


if len(args) < 3:
    raise argparse.ArgumentError("Please, provide all arguments!")
        

# important parameters
gpu_id = 0
layers = 1
neurons = 1000
scale = 2 # 2 - 10
robot_choice = '7DoF-7R-Panda'

# read from path script
#for scale in range(2,12,2):

# build the content of the config file in a dictionary
config_info = {
        'NUM_EXPERIMENT_REPETITIONS': 1,
        'ROBOT_CHOICE': robot_choice,
        'SEED_CHOICE': True,
        'SEED_NUMBER': 0,
        'DEVICE_ID': int(gpu_id),
        'MODEL': {
            'NAME': 'MLP',
            'NUM_HIDDEN_LAYERS': layers,          
            'NUM_HIDDEN_NEURONS': neurons
        },             
        'TRAIN': {
            'DATASET': {
                'NUM_SAMPLES': 10000,
                'JOINT_LIMIT_SCALE': int(scale)
            },
            'CHECKPOINT': {
                'PRETRAINED_G_MODEL': "",
                'RESUMED_G_MODEL': "",
            },
            'HYPERPARAMETERS': {
                'EPOCHS': 10000,
                'BATCH_SIZE': 128,
                'SHUFFLE': True,
                'NUM_WORKERS': 4,
                'PIN_MEMORY': True,
                'PERSISTENT_WORKERS': True,
                'OPTIMIZER_NAME': 'SGD',
                'LEARNING_RATE': 0.0001,
                'BETAS': [0.9, 0.999],
                'EPS': 0.0001,
                'WEIGHT_DECAY': 0.0,
                'WEIGHT_INITIALIZATION': 'default',
                'LOSS': 'l2',
            },
            'PRINT_EPOCHS': True,
            'PRINT_STEPS': 10
        },
}


save_path = "configs/"+robot_choice+"/config_layers_"+str(int(layers))+"_neurons_"+str(int(neurons))
if not os.path.exists(save_path):
            os.makedirs(save_path)

# open a yaml file and dump the content of the dictionary 
with open(save_path+"/train_scale_"+str(int(scale))+".yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Successfully created config files in {}!".format(save_path))