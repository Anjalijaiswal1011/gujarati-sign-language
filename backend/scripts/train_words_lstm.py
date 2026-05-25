import os
import sys

# Add backend directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from services.training_service import _run_training

def main():
    print("=== Training Word LSTM Sequence Recognition Model ===")
    epochs = 20
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Starting LSTM model training for {epochs} epochs...")
    _run_training("words", epochs)
    print("Training finished!")

if __name__ == "__main__":
    main()
