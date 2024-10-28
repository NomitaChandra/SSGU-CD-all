import argparse
import os
import subprocess



# Define the argument parser to get the dataset path from the command line
parser = argparse.ArgumentParser(description="Process CDR dataset")
parser.add_argument('input_path', type=str, help="Path to the dataset directory")
args = parser.parse_args()

# Paths
input_path = args.input_path

# Define the main function to replicate the .sh logic
def main(input_path):
    # Define sets for processing
    sets = ["Test"]

    for d in sets:
        # Run process.py
        process_input_file = os.path.join(input_path, f"CDR_{d}Set.PubTator.txt")
        process_output_file = os.path.join(input_path, d)
        subprocess.run([
            "python3", "/Users/kavithakamarthy/Downloads/SSGU-CD-all/data_processing/process.py",
            "--input_file", process_input_file,
            "--output_file", process_output_file,
            "--data", "CDR"
        ])


    # Rename files
    os.rename(os.path.join(input_path, "Test.data"), os.path.join(input_path, "test.data"))

if __name__ == "__main__":
    main(input_path)
