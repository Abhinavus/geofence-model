import os
import pandas as pd

def create_image_dataset_csv(root_dir, output_csv):
    # Initialize an empty list to store the image paths and labels
    data = []

    # Define the dataset structure (train, test, validation)
    sets = ['train', 'test', 'validation']

    # Loop through each set (train, test, validation)
    for set_name in sets:
        # Loop through cover (0) and steg (1) folders
        for label in ['0', '1']:
            folder_path = os.path.join(root_dir, set_name, label)
            if os.path.exists(folder_path):
                # Loop through each image in the folder
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    if image_path.endswith(('.pgm', '.jpg', '.jpeg', '.bmp')):  # Ensure it's an image file
                        data.append([image_path, int(label)])  # Append image path and label (0 or 1)

    # Create a pandas DataFrame from the list
    df = pd.DataFrame(data, columns=['image_path', 'label'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    print(f"CSV file created: {output_csv}")

# Define the root directory of your dataset
root_dir = "C:\Users\Asus\OneDrive - University of Plymouth\Documents\PROJECT\dataset\UNIWARD_datasets\dataset_pgm_UNIWARD_05"  # Change this to your dataset root directory

# Define the output CSV file path
output_csv = 'steganalysis_dataset.csv'

# Run the function to create the CSV
create_image_dataset_csv(root_dir, output_csv)
