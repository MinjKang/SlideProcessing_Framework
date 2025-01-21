import os
import slideflow as sf
from slideflow.slide import qc
import time

# Define project root and slide directory
project_root = 'data_sample'
slide_dir = '/workspace/1.Raw/WSIs/TCGA_SAMPLE'

# Check if the project already exists
if not os.path.exists(project_root):
    print("Project does not exist. Creating a new project...")
    project = sf.create_project(
        root=project_root,
        slides=slide_dir
    )
else:
    print("Project already exists. Loading the project...")
    project = sf.load_project(project_root)

# Start tile extraction
print("Starting tile extraction...")
start = time.time()

project.extract_tiles(qc=qc.Otsu(), tile_px=512, tile_um=256, save_tiles=True, num_threads=64)

end = time.time()
print("Time taken to extract tiles: ", end - start)
