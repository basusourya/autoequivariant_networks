# Readme

Single augmentation:

For running the augmented FashionMNIST dataset with single augmentations, set aug_array = i and run the following code:

python fashionmnist_classification_hypothesis_testing.py --epochs 10 --aug-array-id i --train-size 10000 --test-size 1000

This will run the tests with augmentation array = one hot vector with ith position = 1. This code iterates through all the single equivariances.






Multiple augmentation:

For running the augmented MNIST dataset with multiple augmentations, set aug_array = i and run the following code:

python fashionmnist_classification_hypothesis_testing.py --epochs 10 --aug-array-id i --train-size 10000 --test-size 1000 --multiple-augmentations

This will run the tests with ith augmentation array taken from Tab.8. This code iterates through all the single equivariances.





Organization of files:

group_transformation_matrices.py: Provides several functions that returns transformed version of matrices according to some predefined groups
equivariance_search_utilities.py: Gets the equivariance parameter sharing indices and provides the equivariant model
equivariance_functions.py: Provides the main algorithm to obtain equivariance parameter sharing indices in Alg.1 in the paper
dataloader.py: Provides dataloader for various datasets with various augmentations
augmentation_functions.py: Provides augmentations functions used in dataloader.py for generating augmented MNIST dataset

