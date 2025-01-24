# SlideProcessing_Framework

## Requirements
- Python >= 3.8
- timm = 1.0.8
- slideflow = 3.0.2
- slideflow-gpl = 0.0.2
- slideflow-noncommercial = 0.0.2

## Data dir (slideflow)
```
data_sample/
  ├─ datasets.json
  ├─ tiles/
  │   └─ 512px_256um/
  │       ├─ SLIDE_1/
  │       │   ├─ 00001.jpg
  │       │   ├─ 00002.jpg
  │       │   └─ ...
  │       ├─ SLIDE_2/
  │       └─ ...
  └─ tfrecords/
```

## usage
1. Tile Extraction
   
```
python extract_tile.py \
    --project_name data_sample \
    --tile_px 512
```


3. Feature Extraction
```
python extract_features.py \
    --project_name data_sample \
    --model_name uni2 \
    --tile_px 512
```


## Reference
Dolezal, J. M., Kochanny, S., Dyer, E., Ramesh, S., Srisuwananukorn, A., Sacco, M., ... & Pearson, A. T. (2024). Slideflow: deep learning for digital histopathology with real-time whole-slide visualization. BMC bioinformatics, 25(1), 134. (https://slideflow.dev/installation/)
https://github.com/slideflow/slideflow-noncommercial
