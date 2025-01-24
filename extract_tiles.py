import slideflow as sf
from slideflow.slide import qc
import argparse
import os

# normalizaton, guassian 옵션 가능하게 추가 예정

# Argument parser 설정
parser = argparse.ArgumentParser(description="Slideflow Tile Extraction Script")
parser.add_argument('--project_name', type=str, default='data_sample', help="Name of the project to create or load")
parser.add_argument('--tile_px', type=int, default=512, help="Tile size in pixels (e.g., 1024, 512, 256)")

args = parser.parse_args()

# 입력 값
project_name = args.project_name
tile_px = args.tile_px

# 현재 코드의 디렉터리 경로를 기준으로 slide_roots 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
slide_roots = os.path.join(current_dir, project_name)

# tile_um 계산 (tile_px에 따라 자동 설정)
tile_conditions = {
    2048: 1024,
    1024: 512,
    512: 256,
    256: 128
}

if tile_px not in tile_conditions:
    raise ValueError(f"Invalid tile_px value: {tile_px}. Must be one of {list(tile_conditions.keys())}.")

tile_um = tile_conditions[tile_px]

if not os.path.exists(project_name):
    print(f"Project '{project_name}' does not exist. Creating a new project...")
    project = sf.create_project(root=project_name, slides=slide_roots)
else:
    print(f"Project '{project_name}' already exists. Loading the project...")
    project = sf.load_project(project_name)

# 타일 추출
print(f"Extracting tiles for project '{project_name}' with tile_px={tile_px}, tile_um={tile_um}...")
project.extract_tiles(qc=qc.Otsu(), tile_px=tile_px, tile_um=tile_um, save_tiles=True, num_threads=64, save_pdf=False)

print("Tile extraction completed.")
