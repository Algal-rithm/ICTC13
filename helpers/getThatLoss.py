import re

# Path to your log file
log_file_path = '/home/researcher/exp1/results/genusResults/noPatch/none_full_test3.txt'
output_file_path = '/home/researcher/exp1/results/genusResults/noPatch/3_fullNone_genus_epochs.txt'

# Create a list to store the extracted data
epoch_data = []

# Regular expression to match the "16/16" part and extract loss/accuracy
log_pattern = r"(\d+)/\d+ \[.*\] -.*loss: (\d+\.\d+) - categorical_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+) - val_categorical_accuracy: (\d+\.\d+)"

# Regular expression to match the "Epoch X/Y" line and extract the epoch number
epoch_pattern = r"Epoch (\d+)/\d+"

# Open the log file and process it
with open(log_file_path, 'r') as f:
    last_epoch = None  # Initialize variable to store the last epoch number
    for line in f:
        # Check if the line matches the epoch pattern
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))  # Capture the current epoch number

        # Check if the line contains the loss and accuracy data
        log_match = re.search(log_pattern, line)
        if log_match and last_epoch is not None:
            epoch_info = {
                'epoch': last_epoch,  # Use the epoch from the previous line
                'loss': float(log_match.group(2)),  # Loss
                'categorical_accuracy': float(log_match.group(3)),  # Categorical accuracy
                'val_loss': float(log_match.group(4)),  # Validation loss
                'val_categorical_accuracy': float(log_match.group(5))  # Validation accuracy
            }
            epoch_data.append(epoch_info)

# Function to save the extracted data to a text file
def save_data_to_file(epoch_data, output_file_path):
    with open(output_file_path, 'w') as f:
        for data in epoch_data:
            f.write(f"Epoch: {data['epoch']} - Loss: {data['loss']}, Accuracy: {data['categorical_accuracy']}, "
                    f"Val Loss: {data['val_loss']}, Val Accuracy: {data['val_categorical_accuracy']}\n")

# Save the extracted data to a text file
save_data_to_file(epoch_data, output_file_path)

# Print a confirmation message
print(f"Data has been saved to {output_file_path}")

