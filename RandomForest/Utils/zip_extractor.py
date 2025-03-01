import zipfile 
import os 
#Define the path to the zip file and the directory to extract to 
zip_file_path = 'C:\\Environment\\Code Files\\MNIST\\digit-recogniz' 
extract_to_path = 'C:\\Environment\\Code Files\\MNIST' 

#Create the directory if it doesn't exist 
os.makedirs(extract_to_path, exist_ok=True) 

#Open the zip file and extract all contents 
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: 
    zip_ref.extractall(extract_to_path) 
    print("Extracted all files to (extract_to_path)")