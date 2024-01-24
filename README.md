# Segment anything with SAM + Grounding DINO + Label Studio

![Segmentation pipeline](static/pipeline-inforgraphic.jpg)

## Table of contents

<!-- TOC -->

## Quickstart

### Step 1: Install dependencies

#### Install Grounding DINO

Follow the tutorial provided by Grounding DINO to install necessary dependencies:  
[Grounding DINO tutorial](https://github.com/IDEA-Research/GroundingDINO).

#### Install SAM

Follow the official SAM tutorial for instalation:  
[SAM tutorial](https://github.com/facebookresearch/segment-anything).

#### Install additional dependencies

Install the remaining dependencies using pip:

```bash
pip install -r requirements.txt
```

### Step 2: Download model weights

Download the model weights for SAM and Grounding DINO. Store them in a directory of your choice (e.g., `weights/`). Remember to update `.gitignore` if you choose a different directory name.

* [SAM weights](https://github.com/facebookresearch/segment-anything#Model-checkpoints)
* [GroundingDINO weights](https://github.com/IDEA-Research/GroundingDINO)

### Step 3: Configure YAML file

Create a configuration file based on the provided template `config_segment_template.yaml`. The following is a minimal configuration example. Detailed configuration information is available in the [Configuration File](#configuration-file) section.

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

### Step 4: Running segmentation

Run the segmentation process using the following command:

```bash
python main.py segment
```

### Step 5: Verify output

Check your root dataset directory for the `metadata.json` file and the `annnotations` directory, which should contain segmentation masks in COCO JSON format.

### Step 6: Segment additional files

To segment new files, simply run the segmentation script again. Use `-c` or `--clean` option to clean annotations if needed.

### Verify segmentation masks

Use the following script to display segmentation masks and their color mapping for the first image:

```python
import matplotlib.pyplot as plt
from segmentation.annotation import AnnotationMetadata

metadata = AnnotationMetadata("./dataset", data_dir="raw_data/")
sample_image = metadata.images[0]

masked_image = sample_image.coco_image.draw_masks()
plt.imshow(masked_image)
plt.show()

# print object:color mapping
print(metadata.colormap())
```

## [Label Studio](https://labelstud.io/) Integration

### Segment anything in Label Studio

To integrate SAM with Label Studio for image segmentation, follow this comprehensive [tutorial](https://labelstud.io/blog/get-started-using-segment-anything/). Ensure that the Label Studio server is configured according to the labeling setup described in the tutorial.

> :bulb: NOTE: The BrushLabels tag name has been updated from "tag" to "brush" in the labeling configuration.

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  <Header value="Brush Labels"/>
  <BrushLabels name="brush" toName="image">
  	<Label value="Dog" background="#FF0000"/>
  	<Label value="Possum" background="#0d14d3"/>
  </BrushLabels>
  <Header value="Keypoint Labels"/>
  <KeyPointLabels name="tag2" toName="image" smart="true">
    <Label value="Dog" smart="true" background="#000000" showInline="true"/>
    <Label value="Possum" smart="true" background="#000000" showInline="true"/>
  </KeyPointLabels>
  <Header value="Rectangle Labels"/>
  <RectangleLabels name="tag3" toName="image" smart="true">
    <Label value="Dog" background="#000000" showInline="true"/>
    <Label value="Possum" background="#000000" showInline="true"/>
  </RectangleLabels>
</View>
```

Ensure that labels correspond to the keys specified in the `ontology`` section of the segmentation phase.

### Exporting annotations to Label Studio

Label Studio uses a proprietary annotation format. Follow these steps to convert COCO JSON annotations for import into Label Studio:

1. Set environmental variables as per the `.env.template`.
2. Create a configuration file for the export process.
3. Upload images to Label Studio project first to ensure matching file names with the exported annotations
4. Run export script.

    ```bash
    IMAGE_SEGMENTATION_CONFIG_PATH=config_export.yaml \
    python main.py export-to-label-studio dataset/annotations/annotations-abc123.json
    ```

    The exported JSON will be saved in the `label_studio_path`` directory, maintaining the same run ID as the original COCO JSON file unless specified otherwise.

5. Import annotations to Label Studio.

### Import annotations from Label Studio

To import modified annotations back into the COCO JSON format.

1. In Label Studio, navigate to the project tab and export annotations as JSON.

2. Save exported project file in a dataset directory, e.g., `label_studio_export`.

3. Update `annotations.export_annotations_path` in the configuration file, e.g., set path to `export/annotations.json`

4. Run parse Label Studio script.

    > :bulb: NOTE: It is recommended to use the same `run_id` as the initial COCO JSON annotations (before being modified in Label Studio). This way, parsed annotation file will have the same name as the initial one, yet will be located in a different directory. Then it is enough to replace `annotations_dir` when initializing `AnnotationMetadata` to use updated labels. 

    ```bash
    FALL_DETECTION_CONTEXT_CONFIG_PATH=config_parse.yaml \
    python main.py parse-label-studio dataset/label_studio_export/project-1.json --coco-path dataset/annotations/annotations-abc123.json --run-id abc123
    ```

5. Imported annotations will be available in the `annotations.export_annotations_path` directory.

## CLI

To use the tool, run the script with desired mode and corresponding arguments. The script supports three modes of operation: `segment`, `parse-label-studio`, and `export-to-label-studio`. The tool uses a [configuration file](#configuration-file) for various settings. It is possible to speify a configuration file through [environmental variables](#environmental-variables).

### Command syntax

```bash
python main.py <mode> [options]
```

### Modes

1. Segment mode

    Segment mode is used for segmenting images and managing annotations.

    **Command**:

    ```bash
    python main.py segment [options]
    ```

    **Options**:

    * `-c`, `--clean`: Clean old annotations and metadata. This action is irreversible.

2. Parse Label Studio mode

    This mode is for parsing exported Label Studio annotations.

    **Command**:

    ```bash
    python main.py parse-label-studio <file> [options]
    ```

    **Arguments**:

    * `file`: Path to the exported Label Studio annotations file.

    **Options**:

    * `--coco-path <path>`: Path to the COCO annotations file for retrieving additional imformation about annotations.

    * `--run-id <id>`: Specify a run UUID. It is recommended to use the same UUID as the COCO annotations file has.

3. Export to Label Studio mode

    This mode is for exporting annotations to the Label Studio format.

    **Command**:

    ```bash
    python main.py export-to-label-studio <coco_annotations_path> [options]
    ```

    **Arguments**:

    * `coco_annotations_path`: Path to the COCO annotations file to be exported.

    **Options**:

    * `--run-id <id>`: Specify UUID.

### Examples

#### Segmenting Images

```bash
python main.py segment --clean
```

## Configuration file

`config_schema.yaml` file contains detailed information about all available parameters for configuration file.

## Environmental variables

### `LABEL_STUDIO_TOKEN`

Token used for authenticating with Label Studio. This token is necessary for operations that require communication with the Label Studio server, such as exporting and importing annotations.

**Required**: Yes (for operations interacting with Label Studio).

**Default value**: None (must be provided by te user).

### `IMAGE_SEGMENTATION_CONFIG_PATH`

Path to the configuration file for the CLI tool. This file contains various settings that the tool uses, such as paths for annotations, logging configurations, and other operational parameters.

**Required**: No (Specify to change the default configuration file)

**Default value**: `config.yaml`

## Dataset Python API

### Overview

The `AnnotationMetadata` class and its descendants in the `annotation.py` file provide a comprehensive API for managing and manipulating image annoations within a [dataset](). This documentation covers the key functionalities of these classes.

### Classes

#### `AnnotationMetadata`

The `AnnotationMetadata` class represents the metadata of a dataset, including images and their annotations.

Initialization

```python
from annotation import AnnotationMetadata

dataset_path = "/path/to/dataset"
metadata = AnnotationMetadata(dataset_path, metadata_filename="metadata.json")
```

Key Methods

*  