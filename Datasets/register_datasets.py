import os

from detectron2.data.datasets import register_coco_instances


def register_dataset(
    dataset_root: str,
    dataset_name: str,
) -> None:
    """
    Register a dataset in Detectron2 format.\n
    We use this to be able to use a modified version of the COCO evaluator already implemented in Detectron2.\n
    The datasets are accessible via the DatasetCatalog and MetadataCatalog.\n
    The names will be dataset_name + "_train" and dataset_name + "_test".\n

    Args:
        dataset_root (str): The root path of the dataset, e.g. C://Users//User//Documents//Datasets
        dataset_name (str): The name of the dataset folder, e.g. "wse2_flakes_instance"
    """
    dataset_path = os.path.join(dataset_root, dataset_name)

    register_coco_instances(
        dataset_name + "_train",
        {},
        os.path.join(dataset_path, "annotations", "train_annotations_300.json"),
        os.path.join(dataset_path, "train_images"),
    )

    register_coco_instances(
        dataset_name + "_test",
        {},
        os.path.join(dataset_path, "annotations", "test_annotations_300.json"),
        os.path.join(dataset_path, "test_images"),
    )


def register_all_datasets():
    dataset_root = os.path.join(os.path.dirname(__file__), "GMMDetectorDatasets")

    for material in ["Graphene", "WSe2"]:
        if os.path.isdir(os.path.join(dataset_root, material)):
            register_dataset(
                dataset_root,
                material,
            )
        else:
            print(f"Dataset {material} not found in {dataset_root}")
