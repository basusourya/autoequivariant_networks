# Readme



For running the neural architecture search algorithm for augmented MNIST with aug_array i run the following code:

python gnas_fcc_MNIST.py --child-epochs 4 --aug-array-id i --child-train-size 4000 --child-test-size 1000

This will run the tests with ith augmentation array taken from Tab.8.





Once the gnas_fcc_MNIST.py returns the top states, use the following code to run them on identical datasets for identical number of epochs as that of single equivariance tests for comparison

python mnist_dqn_accuracies.py --epochs 10 --aug-array-id i --train-size 10000 --test-size 1000 --multiple-augmentations





Organization of files:

gnas_fcc_MNIST.py: Main code for running neural architecture search for augmented MNIST
mnist_dqn_accuracies.py: Trains on datasets and epochs same as single equivariance test for comparison
group_transformation_matrices.py: Provides several functions that returns transformed version of matrices according to some predefined groups
equivariance_search_utilities.py: Gets the equivariance parameter sharing indices and provides the equivariant model
equivariance_functions.py: Provides the main algorithm to obtain equivariance parameter sharing indices in Alg.1 in the paper
dataloader.py: Provides dataloader for various datasets with various augmentations
augmentation_functions.py: Provides augmentations functions used in dataloader.py for generating augmented MNIST dataset

