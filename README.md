# ImageNet-PI

ImageNet-PI is a relabelled version of the standard ILSVRC2012 ImageNet dataset, in which the labels are provided by a collection of 16 deep neural networks with different architectures pre-trained on the standard ILSVRC2012. It also provides privilegd information (PI) about the annotation process in the form of model confidences on each label and meta-data about the networks. The goal of this dataset is to encourage further research on PI and label noise by providing a new large-scale benchmark for research in this area.

## Download instructions

To download the ImageNet-PI dataset, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/google-research-datasets/imagenet_pi
```

2. Install Git LFS to handle the large files:

```bash
git lfs install
```

3. Checkout the main branch:

```bash
cd imagenet_pi
git checkout main
```

4. Pull the data files from Git LFS:

```bash
git lfs pull
```

The dataset is now downloaded and ready to use. You can manually integrate it into your training or use any of our [provided example loaders](#provided-loaders) in `tensorflow/JAX` or `PyTorch`.

## Label and PI generation

Recent work has leveraged Privileged information to make neural networks be more resilient against label noise. Privileged information (or PI for short) refers to features which are available at training time, but missing at test time, such as the features of the annotator that provided the label.

**ImageNet-PI** is a relabelled version of the standard ILSVRC2012 ImageNet dataset in which the labels are provided by a collection of 16 deep neural networks with different architectures pre-trained on the standard ILSVRC2012. Specifically, the pre-trained models are downloaded from [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) and consist of:
- `ResNet50V2`
- `ResNet101V2`
- `ResNet152V2`
- `DenseNet121`
- `DenseNet169`
- `DenseNet201`
- `InceptionResNetV2`
- `InceptionV3`
- `MobileNet`
- `MobileNetV2`
- `MobileNetV3Large`
- `MobileNetV3Small`
- `NASNetMobile`
- `VGG16`
- `VGG19`
- `Xception`

During the re-labelling process, we do not directly assign the maximum confidence prediction of each of the models, but instead, for each example, we sample a random label from the predictive distribution of each model on that example. Furthermore, to regulate the amount of label noise introduced when relabelling the dataset, ImageNet-PI allows the option to use  stochastic temperature-scaling to increase the entropy of the predictive distribution. The stochasticity of this process is controlled by a parameter $\beta$ which controls the inverse scale of a Gamma distribution (with shape parameter $\alpha=1.0$), from which the temperature values are sampled, with a code snippet looking as follows:

```python
# Get the predictive distribution of the model annotator.
pred_dist = model.predict(...)

# Sample the temperature.
temperature = tf.random.gamma(
    [tf.shape(pred_dist)[0]],
    alpha=tf.constant([1.]),
    beta=tf.constant([beta_parameter]))

# Compute the new predictive distribution.
log_probs = tf.math.log(pred_dist) / temperature
new_pred_dist = tf.nn.softmax(log_probs)

# Sample from the new predictive distribution.
class_predictions = tf.random.categorical(tf.math.log(new_pred_dist), 1)[:,0]
```

Intuitively, smaller values of $\beta$ translate to higher temperature values and lead to higher levels of label noise as softmax comes closer to uniform distribution for high temperatures. This re-labelling process can produce arbitrarily noisy labels whose distribution is very far from being symmetrical, i.e., not all mis-classifications are equally likely. For example, it is more likely that similar dog breeds get confused among each other, but less likely that a *dog* gets re-labeled as a *chair*.

The PI in this dataset comes from the confidences of the models on the sampled label, their parameter count, and their test accuracy on the clean test distribution. These PI features are a good proxy for the expected reliability of each of the models.

## Dataset contents

In this release, we provide two standardized sampled annotations obtained by applying the temperature sampling process discussed above: one with $\beta=0.1$ corresponding to **high label noise** and one with $\beta=0.5$ corresponding to **low label noise**. The high-noise version agrees $16.2\%$ of the time with the clean labels and the low-noise version $51.9\%$.

Each version is stored under `high_noise` and `low_noise`, respectively, and consists of the following files:

### `labels-train.csv` and `labels-validation.csv`

These files contain the new (noisy) __labels__ for the training and validation set respectively. The new labels are provided by the pre-trained annotator models. Each file provides the labels in CSV format:

    <image_id>,<confidence_1>,<confidence_2>,...,<confidence_16>

### `confidences-train.csv` and `confidences-validation.csv`

These files contain the confidence of each annotator model in its annotation; both for the training set and the validation set respectively. Each file provides the confidences in CSV format:

    <image_id>,<confidence_1>,<confidence_2>,...,<confidence_16>

### ``annotator-features.csv``
This file contains the annotator features (i.e., meta-data about the annotators themselves) in CSV format (16 rows; one for each model annotator):

    <model_accuracy>,<log_number_of_model_parameters>

The model accuracy is given normalized by the average accuracy of all the models and their standard deviation. The number of parameters are given in a logarithmic scale.

## Provided loaders

ImageNet-PI can be easily loaded using `tensorflow_datasets` for Tensorflow/JAX workflows. We further provide some [example code](./imagenet_pi_torch.py) to load ImageNet-PI as a Pytorch dataset.

## Reference

 If you use ImageNet-PI, please cite this work

 ```bibtex
@article{
        author = {Guillermo Ortiz-Jimenez and Mark Collier and Anant Nawalgaria and Alexander D'Amour and Jesse Berent and Rodolphe Jenatton and Effrosyni Kokiopoulou},
        title = {When does Privileged Information explain away label noise?}
        journal = {arXiv}
        year = {2023}
    }
 ```
