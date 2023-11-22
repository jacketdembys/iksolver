import yaml
import os
import sys
import argparse




# GPUs declaration in the Yaml file
# - NVIDIA-GeForce-GTX-1080-Ti
# - NVIDIA-GeForce-RTX-2080-Ti
# - NVIDIA-A10
# - NVIDIA-GeForce-RTX-3090

# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--layers",
                    type=int,
                    default=1,
                    help="Number of hidden layers.")
parser.add_argument("--neurons",
                    type=int,
                    default=100,
                    help="Number of neurons on each hidden layer.")
parser.add_argument("--scale",
                    type=int,
                    default=2,
                    help="Scale of the joints limits.")
parser.add_argument("--load",
                    type=str,
                    default='cloud',
                    help="local or cloud loading.")


if not len(sys.argv) > 1:
    #raise argparse.ArgumentError("Please, provide all arguments!")
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()


# important parameters
gpu_id = 0
layers = args.layers
neurons = args.neurons
scale = args.scale # 2 - 10
load_option = args.load
robot_choice = '7DoF-7R-Panda'

# read from path script
#for scale in range(2,12,2):

# batch sizes: 4096, 65536
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
                'NUM_SAMPLES': 1000000,
                'JOINT_LIMIT_SCALE': int(scale)
            },
            'CHECKPOINT': {
                'SAVE_OPTIONS': 'cloud',
                'LOAD_OPTIONS': load_option,
                'PRETRAINED_G_MODEL': "",
                'RESUMED_G_MODEL': "",
            },
            'HYPERPARAMETERS': {
                'EPOCHS': 1000,
                'BATCH_SIZE': 250000,
                'SHUFFLE': True,
                'NUM_WORKERS': 4,
                'PIN_MEMORY': True,
                'PERSISTENT_WORKERS': True,
                'OPTIMIZER_NAME': 'Adam',
                'LEARNING_RATE': 0.0001,
                'BETAS': [0.9, 0.999],
                'EPS': 0.0001,
                'WEIGHT_DECAY': 0.0,
                'WEIGHT_INITIALIZATION': 'default',
                'LOSS': 'l2',
            },
            'PRINT_EPOCHS': True,
            'PRINT_STEPS': 100
        },
}


#save_path = "configs/"+robot_choice+"/config_layers_"+str(int(layers))+"_neurons_"+str(int(neurons))+"_scale_"+str(int(scale))
#if not os.path.exists(save_path):
#            os.makedirs(save_path)

# open a yaml file and dump the content of the dictionary 
with open("train.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Successfully created the config file!")