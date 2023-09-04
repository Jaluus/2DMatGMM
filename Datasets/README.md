# Datasets and their Usage

The datasets used in the 2DMatGMM project are currently hosted privately. They will be made available on [Zenodo](https://zenodo.org) soon.

Download links are available here:

- [Graphene GMM Dataset](https://backend.uslu.tech/downloads/data/Graphene.zip)
- [WSe2 GMM Dataset](https://backend.uslu.tech/downloads/data/WSe2.zip)
- [False Positive Dataset](https://backend.uslu.tech/downloads/data/FP_Detector_Dataset.zip)

## GMM Detector Dataset

The GMM Detector Dataset encompasses images of Graphene and WSe<sub>2</sub> material flakes, annotated using the [labelme](https://github.com/wkentaro/labelme) tool.

| Dataset         | Training Images | Testing Images | Annotated Flakes |
| --------------- | --------------- | -------------- | ---------------- |
| Graphene        | 425             | 1362           | 1 to 4 layers    |
| WSe<sub>2</sub> | 92              | 420            | 1 to 3 layers    |

| Material        | Class | Training Instances (> 300 px) | Testing Instances (> 300 px) |
| --------------- | ----- | ----------------------------- | ---------------------------- |
| Graphene        | 1     | 240                           | 938                          |
| Graphene        | 2     | 239                           | 914                          |
| Graphene        | 3     | 191                           | 612                          |
| Graphene        | 4     | 96                            | 494                          |
| WSe<sub>2</sub> | 1     | 76                            | 278                          |
| WSe<sub>2</sub> | 2     | 72                            | 224                          |
| WSe<sub>2</sub> | 3     | 58                            | 171                          |

### Structure

The GMM Detector Dataset should follow this directory structure:

```markdown
GMMDetectorDatasets
├───Graphene
│   ├───annotations
│   ├───test_images
│   ├───test_semantic_masks
│   ├───train_images
│   └───train_semantic_masks
└───WSe2
    ├───annotations
    ├───test_images
    ├───test_semantic_masks
    ├───train_images
    └───train_semantic_masks
```

The annotations folder contains annotation files in the [COCO format](https://cocodataset.org/#format-data), the suffix `_200` indicates the minimum number of pixels used in that file, the annotation files with the `_full` suffix contain all possible flake instances but is not used during evaluation, the evaluation uses the `_300` file.

Furthermore the instances described in the COCO annotation file are transcribed as a semantic mask in the semantic mask folder.

Please note, the provided semantic masks only include instances with an area larger than 200px. Given the images are captured with a 20x Objective, this equates to approximately 30μm² in size.

## False Positive Detector Dataset

The False Positive Detector Dataset consists of masks from a variety of objects detected by the GMM detector. These instances were manually classified as either a true positive or false positive. These annotations are saved in the `annotations.json` file.

The dataset includes 1929 training images and 579 testing images.

| Split | True Positives (`1`) | False Positives (`0`) | Total |
| ----- | -------------------- | --------------------- | ----- |
| Train | 1151                 | 778                   | 1929  |
| Test  | 328                  | 251                   | 579   |

### Structure

The False Positive Detector Dataset should adhere to the following structure:

```markdown
FalsePositiveDetectorDataset
├───train
│   ├───annotations.json
│   ├───masks
│   └───images
└───test
    ├───annotations.json
    ├───masks
    └───images
```

The `annotations.json` file should be structured in such a way that each key is a name of a mask in the `masks` directory without the extension. The value of each key should be either a `0` or a `1` depending on whether the mask is a false positive or not, respectively. An example of the `annotations.json` file is shown below:

```json
{
    "eb62e102-9a1d-4bf4-811e-e5d15c8268db": 1,
    "562851ba-1661-470f-98fe-7f937732e77d": 0,
    ...
    "a8920cb6-f7b7-44e1-a921-14de4cef3049": 0,
}
```

Please note, the `images` directory is only used for visualization purposes and does not factor into the training process.
