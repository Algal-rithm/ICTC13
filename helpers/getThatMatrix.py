import re

# Path to your log file
log_file_path = '/home/researcher/exp1/results/genusResults/patch448/rawData/None_448_test_V3.txt'
output_file_path = '/home/researcher/exp1/results/genusResults/patch448/matrixData/3_448None_genus_matrix.txt'

# Function to preprocess and remove unnecessary formatting
def preprocess_log_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Remove extra newlines, multiple spaces, and tabs. Normalize everything into single spaces.
    content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces, tabs, and newlines with a single space.
    
    # Optionally remove line breaks entirely (making the whole file one large string)
    content = content.replace('\n', ' ').replace('\r', '')

    return content

# Updated regular expression to capture just the confusion matrix
log_pattern = re.compile(r"""
    The\s+stats\s+for\s+(.*?)\s+after\s+(\d+)\s+epochs\s+with\s+(\S+)\s+opt.*?   # Model name, epochs, optimizer
    \[\[(\d+)\s+(\d+)\]\s+\[(\d+)\s+(\d+)\]\]                                  # Confusion matrix cm00, cm01, cm10, cm11
""", re.VERBOSE)

# Process the log file and extract data
log_content = preprocess_log_file(log_file_path)

# Create a list to store the extracted confusion matrices
confusion_matrices = []

# Find all matches in the cleaned content
matches = log_pattern.findall(log_content)

if matches:
    for match in matches:
        # Extract data from the match
        model_name = match[0]
        epochs = int(match[1])
        optimizer = match[2]
        cm_00 = int(match[3])
        cm_01 = int(match[4])
        cm_10 = int(match[5])
        cm_11 = int(match[6])

        # Store the extracted confusion matrix in the list
        confusion_matrices.append({
            'model_name': model_name,
            'epochs': epochs,
            'optimizer': optimizer,
            'confusion_matrix': f"{cm_00} {cm_01} {cm_10} {cm_11}"
        })

# Function to save the extracted confusion matrices to a text file
def save_data_to_file(confusion_matrices, output_file_path):
    with open(output_file_path, 'w') as f:
        for stats in confusion_matrices:
            f.write(f"Model: {stats['model_name']}\n")
            f.write(f"Epochs: {stats['epochs']}\n")
            f.write(f"Optimizer: {stats['optimizer']}\n")
            f.write(f"Confusion Matrix:\n{stats['confusion_matrix']}\n")
            f.write("=" * 50 + "\n")

# Save the extracted confusion matrices to a text file
save_data_to_file(confusion_matrices, output_file_path)

# Print a confirmation message
print(f"Data has been saved to {output_file_path}")


