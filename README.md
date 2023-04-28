# Official Code

Code for "Learning defense transformations for counterattacking adversarial examples"

## Dependencies

Python 3.7.5

Pytorch 1.5.0

cifar2png 0.0.4

torchattacks 1.3


## Datasets

We conduct experiments on CIFAR-10 and CIFAR-100 datasets.

For convenience, we convert CIFAR-10 and CIFAR-100 datasets into PNG images by [cifar2png].


### Train the defense transformer

- Step1: assign the path of cifar10 and convert CIFAR-10 datasets into PNG images.

	- run ``` cifar2png cifar10 datasets/cifar10 ```

- Step2: train classifier h in Algorithm 1.

	- run ``` python train_classifier.py --arch=resnet56 ```

- Step3: generate a set of adversarial examples A in Algorithm 1.

	- run ``` python data_preprocess.py ```

- Step4: train the defense transformer T in Algorithm 1.

	- run ``` python main.py ```


### Evaluation

<!-- Please check the model_h_path and model_ST_path. -->

- Step1: test the defense transformer over ResNet-56 on CIFAR-10.

	- run ``` python evaluation.py --model_h_path 'pytorch_resnet_cifar10/best_model.th' --model_ST_path 'checkpoint_defense_transformer/resnet56_cifar10.pth' ```

- Step2: test the defense transformer over other models on other datasets.
	
	- modify the model and dataset, train the model follow 4 steps above, and run ``` python evaluation.py <model_h_path> <model_ST_path> <dataroot>```