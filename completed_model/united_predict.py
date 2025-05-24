import json
import os
import sys
from pathlib import Path
import threading

# Importing Necessary Components from the Original File (Correcting the Import Path)
sys.path.append('.')  # Add the current directory to the system path
from rgb_predict import predict_and_save_rgb
from normal_predict import predict_and_save_normal
from point_cloud_predict import predict_and_save_point_cloud

def load_predictions(file_path):
    """Load the prediction results"""
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found.")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_united_prob(rgb, normal, point):
    """Calculate joint probabilities"""
    united = {}
    all_ids = set(rgb.keys()) | set(normal.keys()) | set(point.keys())
    for cid in all_ids:
        # Unified category formatting, remove extensions
        cid_base = cid.split('.')[0]
        p_rgb = rgb.get(cid_base, 0)
        p_normal = normal.get(cid_base, 0)
        p_point = point.get(cid_base, 0)
        p_united = p_rgb*0.45 + p_normal*0.45 + p_point*0.1
        united[cid_base] = p_united
    return united

def save_json(data, file_path):
    """Save the result to a JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f)

def united_predict():
    # Create a save directory
    save_dir = r'.\saved_files'
    os.makedirs(save_dir, exist_ok=True)

    # Define the JSON file path
    rgb_json = Path(save_dir) / 'rgb_predictions.json'
    normal_json = Path(save_dir) / 'normal_predictions.json'
    point_cloud_json = Path(save_dir) / 'point_cloud_predictions.json'

    # Define a list of threads
    threads = []

    # Define thread functions
    def run_and_save_rgb():
        results = predict_and_save_rgb()
        save_json(results, rgb_json)

    def run_and_save_normal():
        results = predict_and_save_normal()
        save_json(results, normal_json)

    def run_and_save_point_cloud():
        results = predict_and_save_point_cloud()
        save_json(results, point_cloud_json)

    # Create and start a thread
    threads.append(threading.Thread(target=run_and_save_rgb))
    threads.append(threading.Thread(target=run_and_save_normal))
    threads.append(threading.Thread(target=run_and_save_point_cloud))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Load the prediction results
    rgb_results = load_predictions(rgb_json)
    normal_results = load_predictions(normal_json)
    point_cloud_results = load_predictions(point_cloud_json)

    if rgb_results is None or normal_results is None or point_cloud_results is None:
        return

    # Calculate joint probabilities
    united_probs = calculate_united_prob(rgb_results, normal_results, point_cloud_results)

    # Get the top 5
    top5 = sorted(united_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 United Predictions:")
    for i, (cid, prob) in enumerate(top5, 1):
        print(f"{i}. {cid}: {prob * 100:.18f}%")

if __name__ == "__main__":
    united_predict()