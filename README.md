# ImageNet-PI

Recent work has leveraged Privileged information to make neural networks be more resilient against label noise. Privileged information (or PI for short) refers to features which are available at training time, but missing at test time, such as the features of the annotator that provided the label.

**ImageNet-PI** is a relabelled version of the standard ILSVRC2012 ImageNet dataset in which the labels are provided by a collection of 16 deep neural networks with different architectures pre-trained on the standard ILSVRC2012. Specifically, the pre-trained models are downloaded from [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

During the relabelling process, we do not directly assign the maximum confidence prediction of each of the models, but instead, for each example, we sample a random label from the predictive distribution of each model on that example. Furthermore, to regulate the amount of label noise introduced when relabelling the dataset, ImageNet-PI allows the option to use some stochastic temperature-scaling to increase the entropy of the predictive distribution. The stochasticity of this process is controlled by a parameter β which controls the inverse scale of a Gamma distribution, from which the temperature values are sampled. Intuitively, smaller values of beta translate to larger levels of label noise. This re-labelling process can produce arbitrarily noisy labels whose distribution is very far from symmetrical, i.e., not all misclassifications are equally likely. For example, it is more likely that similar dog breeds get confused among each other, but less likely that a dog gets relabeled as a chair. 

The PI in this dataset comes from the confidences of the models on the sampled label, their parameter count, and their test accuracy on the clean test distribution. These PI features are a good proxy for the expected reliability of each of the models.

## Description of the files

### labels-train.csv and labels-validation.csv
These files contain the new (noisy) __labels__ for the training and validation set respectively. The new labels are provided by the pre-trained annotator models. Each file provides the labels in CSV format:

    <image_id>,<confidence_1>,<confidence_2>,...,<confidence_16>
    
### confidences-train.csv and confidences-validation.csv
These files contain the confidence of each annotator model in its annotation; both for the training set and the validation set respectively. Each file provides the confidences in CSV format:

    <image_id>,<confidence_1>,<confidence_2>,...,<confidence_16>
 
### annotator-features.csv
This file contains the annotator features (i.e., meta-data about the annotators themselves) in CSV format (16 rows; one for each model annotator):

    <model_accuracy>,<number_of_model_parameters>
 
