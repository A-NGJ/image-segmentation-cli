# Segment anything with SAM + Grounding DINO + Label Studio

![Segmentation pipeline](static/pipeline-inforgraphic.jpg)

## Table of contents

1. [Quickstart](#quickstart)
2. [Label Studio Integration](#label-studio-integration)
3. [CLI](#cli)
4. [Configuration File](#configuration-file)
5. [Environmental Variables](#environmental-variables)
6. [Fall dataset](#fall-dataset)
7. [Dataset Python API](#dataset-python-api)
    - [Classes](#classes)
    - [API examples](#api-examples)
    - [Transformations](#transformations)
    - [Pytorch dataset](#pytorch-dataset)

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

### CLI Examples

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

## Fall dataset

We release a fall dataset that was annotated using our image segmentation CLI. It consists of frames extracted from public fall datasets: [KULeuven](https://iiw.kuleuven.be/onderzoek/advise/datasets#High%20Quality%20Fall%20Simulation%20Data) [1], [UR Fall](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) [2], and [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky/4) [3].

:arrow_down: The dataset is available [here](). :arrow_down:

| Name      | No. falls | % Falls | No. non-falls | % Non-falls | Total | % Total |
|-----------|-----------|---------|---------------|-------------|-------|---------|
| CAUCAFall | 1,538     | 67      | 1,575         | 41          | 3,113 | 51      |
| KULeuven  | 713       | 31      | 1,950         | 51          | 2,663 | 44      |
| URFall    | 42        | 2       | 275           | 8           | 317   | 5       |
| **Total** | **2,293** | **100** | **3,800**     | **100**     | **6,093** | **100** |

*Table: Comparative analysis of fall and non-fall data across subsets of the merged dataset. The table presents the number of fall (No. falls) and non-fall (No. non-falls) instances along with their respective percentage shares (% Falls and % Non-falls) for each dataset. "Total" columns refer to the combined count of fall and non-fall instances, whilst "Total" row refers to the sum of a respective column.*


[1] Greet Baldewijns et al. “Bridging the gap between reallife data and simulated data
by providing a highly realistic fall dataset for evaluating camerabased fall detection
algorithms”. In: Healthcare Technology Letters 3.1 (2016), pp. 6–11. DOI: 10.1049/
htl.2015.0047.

[2] Bogdan Kwolek and Michal Kepski. “Human fall detection on embedded platform
using depth maps and wireless accelerometer”. In: Computer Methods and Pro
grams in Biomedicine 117.3 (Dec. 2014), pp. 489–501. ISSN: 01692607.

[3] Jose Camilo Eraso et al. Dataset CAUCAFall. Version V4. 2022. DOI: 10.17632/
7w7fccy7ky.4.

## Dataset Python API

### Overview

The `AnnotationMetadata` class and its descendants in the `annotation.py` file provide a comprehensive API for accessing and managing image annoations. This documentation covers the key functionalities of these classes.

### Classes

#### `AnnotationMetadata`

The `AnnotationMetadata` class represents the metadata of a dataset, including images and their annotations.

Initialization

```python
from annotation import AnnotationMetadata

dataset_path = "/path/to/dataset"
metadata = AnnotationMetadata(dataset_path)
```

Key Methods

* `filter_images`: Filters images based on specific criteria.

    Criteria uses class attributes as dictionary keys, and filter function as values.
    A filter returns all images for which a filter function returns true. For instance, in the first example all images which `coco_image.file_name` attribute is equal to `img_name` are returned, effectively retrieving one specific image.

    ```python
    img_name = "img1.png"
    sample_img = metadata.filter_images(
        {"coco_image.file_name": lambda x: x == img_name}
    )[0]

    # Filter images by label(s)
    fall_images = metadata.filter_images({
        "labels.fall": lambda x: x == "fall",
        "labels.set": lambda x: x == "train",
    })
    ```

* `to_json`: Saves updated metadata to a JSON file.

  ```python
  metadata.to_json("updated_metadata.json")
  ```

* `remove_images`: Remove images that match supplied filters. Filters are defined the same way as in `filter_images` method.
* `merge`: Merge two annotation metadata files.
* `get_filenames`: Returns names of images in the dataset.
* `get_img_index`: Get indices of filenames of segmented images from a metadata file.
* `get_labels`: Get dictionary of labels for each image, where key is the image file name.

Key attributes

* `images` List[AnnotationMetaImage]: List of annotations of all images in the dataset.
* `image_labels`: Dict[str, List[str]]: A dictionary of each label key and its possible valyes.
* `categories`: Dict[str, int]: A dictionary of <category_name>:<category_id> pairs.

#### `AnnotationMetaImage`

The class representes metadata for a single image in the dataset, including its annotations.

Usage

```python
annotation_meta_image = metadata.images[0]
```

Key attrbiutes

* `annotation_file`: str: A name of the metadata annotation file.
* `coco_image`: CocoImage: COCO-style annotated image.
* `labels`: Dict[str, str]: Image labels (excluding segmentation masks)

#### `CocoImage`

Handles operations specific to COCO-style annotated images.

Key Methods

* `mask_objects`: Apply transformation to chosen objects in an image
* `draw_masks`: Draws all annotation masks on the image.

    ```python
    import matplotlib.pyplot as plt

    masked_image = annotation_meta_image.coco_image.draw_masks()
    plt.imshow()
    plt.show()
    ```

* `get_bbox`: Get bounding box coordinates for a given annotation.

Key attributes

* `file_name`: str: Image file name.
* `width`: int: Image width.
* `height`: int: Image height.
* `image`: numpy.ndarray: Binary image
* `annotations`: List[SingleCocoAnnotation]: Annotations containing bounding boxes and segmentation masks of detected objects in the image.

#### `SingleCocoAnnotation`

Key attributes

* `rle`: List[int]: Run Length Encoding mask.
* `bbox`: List[int]: Bounding box coordinates.
* `category_id`: int
* `category`: CocoCategory: object containing category id and name

### API Examples

#### Moving image between sets

```python
# Move images to a different set (e.g., from training to test set)
for img in fall_images:
    img.labels["set"] = "test"
```

#### Filtering by annotation

```python
# Filter images containing a specific annotation
wheelchair_images = metadata.filter_images({
    "coco_image.annotations": lambda annotations: "wheelchair" in 
    [a.category.name for a in annotations]
})

```

### Transformations

The `segmentation/transformations.py` module contains transformations that can be directly applied on segmented objects.

Available transformations:

#### `DummyTransformation()`

An identity transformation that makes no changes to the object. Useful when no transformation is desired (at least one transformation is always required).

#### `BlurTransformation(kernel_size=15, inverse=False, probability=1)`

Apply Gaussian blur with chosen kernel size and probability.

**Parameters**

* `kernel_size`: The size of the Gaussian kernel. Default is 15.
* `inverse`: If `True`, applies the transformation to the area outside the segmented object. Default is`False`.
* `probability`: The probability of applying the transformation. Default is 1.

#### `SolidColorTransformation(color=(0, 0, 0), inverse=False)`

Change pixel color to a solid color defined in as RGB values.

**Parameters**

* `color`: : A tuple representing the RGB values of the desired color. Default is (0, 0, 0) (black).
* `inverse`: If `True`, applies the transformation to the area outside the segmented object. Default is `False`.

#### `TorchTransformation(transformation, inverse=False, **kwargs)`

Apply a transformation from `torchvision` package. Currently available: `RandAugment`, `AutoAugment`, `ColorJitter`, `GrayScale`.

**Parameters**

* `transformation`: The torchvision transformation to apply (e.g., `RandAugment`, `AutoAugment`, `ColorJitter`, `GrayScale`).
* `inverse`: If `True`, applies the transformation to the area outside the segmented object. Default is `False`.
* `kwargs`: Additional keyword arguments specific to the chosen torchvision transformation.

You can define any transformation as long as it implements base `Transformation`.

The following example demonstrates how to apply a blur transformation to the inverse of a segmented object labeled as "person" and then display the modified image:

```python
from segmentation.transformation import BlurTransformation

masked_image, mask = image.coco_image.mask_objects(
  ["person"], mask_transformation=BlurTransformation(inverse=True),
)

plt.imshow(masked_image)
plt.show()
```

### Pytorch Dataset

The `SegmentationDataset` class from the `segmentation/data.py` module is a custom PyTorch Dataset implementation for image segmentation tasks, fully compatible with `AnnotationMetadata`. It integrates efficiently with PyTorch's data handling and allows for the application of complex image transformations on segmented objects.

Initialization

```python
from segmentation import SegmentationDataset, DummyTransformation
from segmentation.annotation import AnnotationMetadata

dataset = SegmentationDataset(
    metadata=AnnotationMetadata("/path/to/dataset"),
    segmentation_transforms=[DummyTransformation()],
    objects=[["person"]],
    transforms=None,  # Optional standard PyTorch transforms that are applied after segmentation transforms
    target_label="your_target_label"
)
```

**Parameters**

* `metadata`: `AnnotationMetadata`: An instance of `AnnotationMetadata` containing dataset annotations.
* `segmentation_transforms`: Sequence[Transformation]: A sequence of segmentation transformations to be applied.
* `objects`: Sequence[Sequence[str]]: A sequence of object groups. Each group corresponds to a transformation in segmentation_transforms.
* `transforms`: Optional[Callable]: Standard PyTorch transformations to be applied to the entire image.
* `target_label`: str: The label of interest from the metadata.

**Usage example**

The following example demonstrates how to use the SegmentationDataset with a simple transformation and PyTorch's DataLoader:

```python
from torch.utils.data import DataLoader
from segmentation.transformations import BlurTransformation

# Initialize the dataset
dataset = SegmentationDataset(...)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over the DataLoader
for images, labels, file_names in data_loader:
    # Process images and labels
    pass
```
