import os
import glob
import argparse
import torch
import timm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from shutil import copyfile
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import slideflow as sf
from slideflow.slide import qc

###############################################################################
# 1) HF 모델 가중치 로드
###############################################################################
def load_weights_from_hf(hf_model_map, model_name, device):

    if model_name not in hf_model_map:
        raise ValueError(f"Model {model_name} not supported for Hugging Face download.")

    print(f"[Info] Downloading weights for {model_name} from Hugging Face...")

    model = timm.create_model(hf_model_map[model_name], pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model = model.to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()

    return model, transform

###############################################################################
# 2) ImageDataset (PNG 이미지 로딩 + Transform)
###############################################################################
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            'input': image,
            'input_path': img_path
        }

###############################################################################
# 3) 타일 디렉토리에서 PNG를 읽어 model inference → CSV 저장
###############################################################################
def save_feature_vectors(image_folder, model, transform, save_folder, device='cuda'):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # image_folder 내부의 하위 디렉토리(슬라이드 단위)를 훑음
    subfolders = sorted(os.listdir(image_folder))

    for index, subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(image_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        csv_save_path = os.path.join(save_folder, f'{subfolder}.csv')

        if os.path.exists(csv_save_path):
            print(f'[Skip] {subfolder_path} → CSV already exists.')
            continue

        print(f'[Process {index+1}/{len(subfolders)}] {subfolder_path}')

        jpg_list = glob.glob(os.path.join(subfolder_path, '*.jpg'))
        if len(jpg_list) == 0:
            print(f'No .jpg files in {subfolder_path}, skipping.')
            continue

        print(f'Found {len(jpg_list)} images in {subfolder_path}')

        # DataLoader 구성
        dataset = ImageDataset(jpg_list, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        feature_vectors = []
        image_names = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                img = batch['input'].to(device)
                paths = batch['input_path']

                # 모델 forward → 임베딩 추출
                feats = model(img)
                feat_np = feats.cpu().numpy()

                feature_vectors.extend(feat_np)
                image_names.extend([os.path.basename(p) for p in paths])

        # CSV 저장
        if feature_vectors:
            df = pd.DataFrame(feature_vectors)
            df.index = image_names
            df.columns = [str(i) for i in range(df.shape[1])]

            print(f"Saving CSV: {csv_save_path} (rows={len(df)})")
            df.to_csv(csv_save_path, index=True, header=True)
        else:
            print(f"[Warn] No features extracted for {subfolder_path}.")

###############################################################################
# 4) 메인 함수: SlideFlow 프로젝트 로드 + TIMM 모델 + 특징 추출
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="SlideFlow + TIMM Feature Extraction")
    parser.add_argument('--project_name', type=str, default='data_sample',
                        help="SlideFlow project name/path")
    parser.add_argument('--model_name', type=str, default='uni2',
                        help="Hugging Face model (uni or virchow)")
    parser.add_argument('--tile_px', type=int, default=512,
                        help="Tile size in pixels (e.g. 512, 1024)")
    args = parser.parse_args()

    ###########################################################################
    # A. SlideFlow 프로젝트 로드
    ###########################################################################
    project = sf.load_project(args.project_name)

    hf_model_map = {
        'uni2': "hf-hub:MahmoodLab/UNI2-h",
        'uni': "hf-hub:MahmoodLab/UNI",
        'virchow': "hf_hub:paige-ai/Virchow2"
    }

    # tile_um = 256, 128 등 슬라이드 프로젝트별 규칙에 맞게 매핑
    tile_conditions = {
        2048: 1024,
        1024: 512,
        512: 256,
        256: 128
    }

    if args.tile_px not in tile_conditions:
        raise ValueError(f"Invalid tile_px={args.tile_px}, must be in {list(tile_conditions.keys())}")

    dataset = project.dataset(tile_px=args.tile_px, tile_um=tile_conditions[args.tile_px])
    print("\n[Info] SlideFlow project and dataset loaded.")
    print(dataset.summary())
    
    label_key = f"{args.tile_px}px_{tile_conditions[args.tile_px]}um"
    tiles_base = dataset.sources["MyProject"]["tiles"]

    image_folder = os.path.join(tiles_base, label_key)
    save_dir = os.path.join(args.project_name, "features", args.model_name, label_key)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Info] Using tile folder: {image_folder}")
    
    ###########################################################################
    # B. 모델 로드 (Hugging Face -> local weights -> timm)
    ###########################################################################
    device_id = 6
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    model, transform = load_weights_from_hf(hf_model_map, args.model_name, device)


    ###########################################################################
    # C. 타일 폴더에서 PNG → 특징 추출 → CSV 저장
    ###########################################################################
    save_feature_vectors(
        image_folder=image_folder,
        model=model,
        transform=transform,
        save_folder=save_dir,
        device=device
    )

    print("\n[Done] Feature extraction completed.")

if __name__ == "__main__":
    main()
