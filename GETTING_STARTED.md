# Getting Started with 2DMatGMM

Welcome to the guide on getting started with the 2DMatGMM Model. This Guide provides a comprehensive overview of how to use the model, from setting up an inference demo with pre-trained weights, to training and evaluation.  

For a quick demo you can run the `demo/detect_flakes.ipynb` notebook to see the model in action and find out how to use and customize it.

Before running the `demo/demo.py` as well as the Training and Evaluation scripts, please make sure to download the datasets.
See our [Dataset Guide](./Datasets/README.md) for detailed information.

## Running the Inference Demo with Pre-trained Weights

To showcase the capabilities of the 2DMatGMM, a `demo/demo.py` script has been provided.
When running the script the detector will be run on the test set of the downloaded datasets and store the visualizations in the defined output directory.  
Execute this script using the following command:

```shell
python demo/demo.py --material Graphene
```

There are several options available to customize the execution of the script:

* `--out`: Specifies the output directory to save visualizations. Default is `Output`.
* `--num_image`: Defines the number of images to process. Default is `10`.
* `--material`: Selects the material for inference. Default is `Graphene`. Other options include `WSe2`.
* `--size`: Specifies the size threshold in pixels. Default size is `200` pixels.
* `--std`: Sets the standard deviation threshold. Default value is `5`, as referenced in the paper.
* `--min_confidence`: Adjusts the minimum required confidence to visualize flakes, with confidence defined as (1-FP_Probability). Default value is `0.5`.
* `--channels`: Determines the channels to use. Default is `BGR`. This option is mostly ignorable for the purposes of the demo.
* `--shuffel`: Enables selection of random images from the test set instead of the same ones. Default value is `False`.

For example, to run the inference demo on 20 'WSe2' images, saving visualizations in 'WSe2_Outputs' directory, with a size threshold of 300 pixels, a standard deviation threshold of 6, a minimum confidence of 0, and shuffling enabled, use the following command:

```shell
python demo.py --out WSe2_Outputs --num_image 20 --material WSe2 --size 300 --std 6 --min_confidence 0 --shuffel True
```


## Training the Model using your own Data

To interactively train the model using your own data, we have provided a Jupyter Notebook `Interactive_Parameter_Estimation.ipynb` in the `GMMDetector` folder.
You can follow the instructions in the notebook to train the model using your own data.

## Training the Model using the Paper Dataset

We have included two scripts to independently train the Gaussian Mixture Model and the False Positive Detector.  
To train the GMM on the Train dataset from the Paper, use this command:

```shell
python GMMDetector/train_parameters.py
```

The above command will train the GMM on both Graphene and WSe2 datasets. The existing trained weights in `GMMDetector/trained_parameters` will be replaced.

For training the False Positive Detector using the Dataset from the Paper, use:

```shell
python FalsePositiveDetector/train_false_positive_detector.py
```

The existing model saved in `FalsePositiveDetector/models` will be overwritten.
To retain the previous trained models, consider renaming them or moving them to a different directory.

## Evaluating the Model

We provide two scripts, `evaluate_instance_metrics.py` and `evaluate_semantic_metrics.py`, to evaluate the model and reproduce the metrics from the paper.
Before running the evaluation, the Detection2 Package needs to be installed.  
Our [Installation Guide](INSTALL.md) offers detailed steps for this installation.  
To evaluate the instance metrics, use:

```shell
python evaluate_instance_metrics.py
```

This command will take approximately 10 minutes to execute and will save the results in the `Metrics` folder.
To evaluating the semantic metrics, execute:

```shell
python evaluate_semantic_metrics.py
```

This evaluation will require more time - approximately 1 to 2 hours - as the model is evaluated across all possible confidence thresholds. The results are saved as a plot in the `Metrics` folder.
