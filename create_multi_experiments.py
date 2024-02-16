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
parser.add_argument("--jvar",
                    type=int,
                    default=1,
                    help="Joint variation when generating dataset.")
parser.add_argument("--blocks",
                    type=int,
                    default=5,
                    help="Number of blocks if ResMLP or DenseMLP.")
parser.add_argument("--load",
                    type=str,
                    default='cloud',
                    help="local or cloud loading.")
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="seed choice.")


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
num_blocks = args.blocks
joint_variation = args.jvar
seed_choice = args.seed
robot_choice = '7DoF-GP66'   #'7DoF-7R-Panda' '7DoF-GP66' 

# read from path script
for joint_variation in range(1,21):
#for scale in range(2,12,2):
    neuron = 1024
#for neuron in range(128, neurons+128, 128):

    # batch sizes: 4096, 65536
    # build the content of the config file in a dictionary
    config_info = {
        'NUM_EXPERIMENT_REPETITIONS': int(seed_choice),
        'ROBOT_CHOICE': robot_choice,
        'SEED_CHOICE': True,
        'SEED_NUMBER': int(seed_choice),
        'DEVICE_ID': int(gpu_id),
        'MODEL': {
            'NAME': 'DenseMLP3',      # MLP, ResMLP, DenseMLP3, DenseMLP 
            'NUM_HIDDEN_LAYERS': layers,          
            'NUM_HIDDEN_NEURONS': neurons,
            'NUM_BLOCKS': num_blocks
        },             
        'TRAIN': {
            'DATASET': {
                'NUM_SAMPLES': 1000000,
                'JOINT_LIMIT_SCALE': int(scale),
                'JOINT_VARIATION': int(joint_variation),
                'TYPE':'seq', # 1_to_1, seq
                'ORIENTATION': 'RPY' # RPY, Quaternion, DualQuaternion, Rotation, Rotation6d
            },
            'CHECKPOINT': {
                'SAVE_OPTIONS': 'cloud', # local, cloud
                'LOAD_OPTIONS': load_option,
                'PRETRAINED_G_MODEL': "",
                'RESUMED_G_MODEL': "",
            },
            'HYPERPARAMETERS': {
                'EPOCHS': 1000,
                'BATCH_SIZE': 128, #100000
                'SHUFFLE': True,
                'NUM_WORKERS': 4,
                'PIN_MEMORY': False,
                'PERSISTENT_WORKERS': True,
                'OPTIMIZER_NAME': 'Adam', # Adam, SGD
                'LEARNING_RATE': 1e-3, #0.0001, # MLP / RMLP -> 0.001 and DMLP -> 0.0001
                'BETAS': [0.9, 0.999],
                'EPS': 0.00001,
                'WEIGHT_DECAY': 0.0,
                'WEIGHT_INITIALIZATION': 'default',
                'LOSS': 'lq',           # lq, ld
            },
            'PRINT_EPOCHS': True,
            'PRINT_STEPS': 100
        },
    }


    #save_path = "configs/"+robot_choice+"/config_layers_"+str(int(layers))+"_neurons_"+str(int(neurons))+"_scale_"+str(int(scale))
    #if not os.path.exists(save_path):
    #            os.makedirs(save_path)

    # open a yaml file and dump the content of the dictionary 
    with open("train_"+str(joint_variation)+".yaml", 'w') as yamlfile:
        data = yaml.dump(config_info, yamlfile)
        print("Successfully created the config file!")