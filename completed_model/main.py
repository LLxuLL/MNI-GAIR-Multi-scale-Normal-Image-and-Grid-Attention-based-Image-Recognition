import os
import threading
import torch
import warnings

warnings.filterwarnings("ignore")

# Define the training function
def train_model(script_path, model_name):
    os.system(f'python {script_path}')
    print(f'{model_name} training completed!')

# Define the merge weight function
def merge_weights(normal_path, rgb_path, point_cloud_path, merged_path):
    # Loads the weights of each model
    normal_weights = torch.load(normal_path)
    rgb_weights = torch.load(rgb_path)
    point_cloud_weights = torch.load(point_cloud_path)

    # Merge weights
    merged_weights = {
        'normal': normal_weights,
        'rgb': rgb_weights,
        'point_cloud': point_cloud_weights
    }

    # Save the merged weights
    torch.save(merged_weights, merged_path)
    print('Merged model weights saved to', merged_path)

# Create a thread
threads = []

# Add normal_train.py threads
thread = threading.Thread(target=train_model, args=('normal_train.py', 'Normal Model'))
threads.append(thread)
thread.start()

# Add rgb_train.py threads
thread = threading.Thread(target=train_model, args=('rgb_train.py', 'RGB Model'))
threads.append(thread)
thread.start()

# Add point_cloud_train.py threads
thread = threading.Thread(target=train_model, args=('point_cloud_train.py', 'Point Cloud Model'))
threads.append(thread)
thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Merge weights
merge_weights(
    r'.\saved_models\normal_facescape_model.pth',
    r'.\saved_models\rgb_facescape_model.pth',
    r'.\saved_models\point_cloud_facescape_model.pth',
    r'.\saved_models\merged_facescape_model.pth'
)