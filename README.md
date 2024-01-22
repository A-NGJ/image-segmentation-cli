# Segment anything with SAM + Grounding DINO

## Table of contents

## Quickstart

1. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

    To install Grounding DINO, follow their [tutorial](https://github.com/IDEA-Research/GroundingDINO).

2. Download SAM and GroundingDINO weights

    * [SAM weights](https://github.com/facebookresearch/segment-anything#Model-checkpoints)
    * [GroundingDINO weights](https://github.com/IDEA-Research/GroundingDINO)

    Download the weights to a directory of your choice (`weights/` is a good default). If you choose a different name, remember to add it to `.gitignore`, because it is not recommended to upload large binary files to version control.

3. Specify YAML configuration

The following snippet is a minimal configuration that can be used to run segmentation. Use `config_segment_template.yaml` as a boilerplate. Detailed config information can be found in the [Config](#config) section.

```yaml
grounding_dino:
  config_path: /path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
  checkpoint_path: /path/to/groundingdino_swint_ogc.pth
  box_threshold: 0.4
  text_threshold: 0.25
sam:
  checkpoint_path: /path/to/sam_vit_h_4b8939.pth
  encoder_version: vit_h
checkpoint_step: 100
ontology:
  key:
    - "person"
  value:
    - "a human being"
data:
  image_dir: ./raw_data
  image_extensions: [".jpg", ".png", ".jpeg"]
annotations:
  min_image_area_percentage: 0.002
  max_image_area_percentage: 0.8
  approximation_percentage: 0.75
  root_path: ./dataset
  annotations_path: annotations/annotations.json
  labels:
    set: ["train", "val", "test"]
```

## Config
