import os
from contextlib import closing
from zipfile import ZipFile


# --- Read all Zip files in Sima-Dataset Folder 
directory= '../cleaned-dataset'
files_in_directory = os.listdir(directory)
filtered_files = [file for file in files_in_directory if file.endswith(".zip")]

# --- Core
total_count = 0
for file in filtered_files:
    # --- Get file
    path_file = os.path.join(directory, file)
    with closing(ZipFile(path_file)) as archive:
        count = len(archive.infolist())
        total_count += count
        print(path_file ,count)

print("Total face images: ", total_count)