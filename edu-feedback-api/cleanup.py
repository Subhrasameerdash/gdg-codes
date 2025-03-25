import os
import shutil

def remove_unnecessary_files(root_dir='.'):
    # List of directories to remove
    folders_to_remove = ['.ipynb_checkpoints', '__pycache__']

    # Extensions for temporary/cache files
    temp_file_extensions = ['.tmp', '.log', '.cache', '.bak']

    # Walk through all files and directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove specified folders
        for folder in folders_to_remove:
            folder_path = os.path.join(dirpath, folder)
            if os.path.exists(folder_path):
                print(f"Removing folder: {folder_path}")
                shutil.rmtree(folder_path)

        # Remove temporary/cache files
        for filename in filenames:
            if any(filename.endswith(ext) for ext in temp_file_extensions):
                file_path = os.path.join(dirpath, filename)
                print(f"Removing file: {file_path}")
                os.remove(file_path)

# Run the cleanup
remove_unnecessary_files()
print("Cleanup completed successfully!")
