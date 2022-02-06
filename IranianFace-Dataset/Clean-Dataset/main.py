import os
import zipfile
from cv2 import cv2
from os.path import basename

# --- Define all functions
def make_zip(name, dirName):
    # --- create a ZipFile object
    with zipfile.ZipFile(name, 'w') as zipObj:
        # --- Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirName):
            for filename in filenames:
                # --- create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # --- Add file to zip
                zipObj.write(filePath, basename(filePath))


# --- Read all Zip files in Sima-Dataset Folder 
directory= '../'
files_in_directory = os.listdir(directory)
filtered_files = [file for file in files_in_directory if file.endswith(".zip")]

# --- Core
for file in filtered_files[65::]:
    # --- Get file
    path_file = os.path.join(directory, file)
    print("File name: ", path_file, " Has been Started!!!!!!!!!")

    # --- Unzip file
    with zipfile.ZipFile(path_file,"r") as zip_ref:
        zip_ref.extractall(path_file[:-4])

    # --- Read face images from Folder
    faces = os.listdir(path_file[:-4])
    for file in faces:
        # --- get face addr
        path_file_new = os.path.join(path_file[:-4], file)
        # --- read face
        img = cv2.imread(path_file_new)
        cv2.imshow(path_file[:-4], img)
        # --- wait for kill or save
        key = cv2.waitKey(0)
        print(key)
        if key == 107:
            os.remove(path_file_new)
            print("face has been removed!! : ", path_file_new)
        else:
            continue
    
    # --- Make zip file
    make_zip("../cleaned-dataset/"+path_file[3::], path_file[:-4])
