import os
import subprocess

def run_data_utils():
    print('Generating training data...')
    subprocess.run(['python', 'data_utils.py'], check=True)
def run_train():
    print('Training the model...')
    subprocess.run(['python', 'train.py'], check=True)
def run_evaluate():
    print('Evaluating the model...')
    subprocess.run(['python', 'evaluate.py'], check=True)

if __name__ == "__main__":
    run_data_utils()
    run_train()
    run_evaluate()