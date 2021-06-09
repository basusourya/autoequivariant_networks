# Readme


Steps for running DQN algorithm:
- Go to GrouPy folder (https://github.com/basusourya/GrouPy) (used from publically available code by Cohen and Welling https://github.com/tscohen/GrouPy and slightly modified as was required)
- run the following for dataset named "X"

python setup.py install
python gnas_GCNN_X.py --Q-GAMMA 0.5 --Q-BATCH_SIZE 256 --max_episodes 200 --max_models 600

Replace X by the datasets CIFAR10, SVHN, RotMNIST, ASL, KMNIST, EMNIST as required


Once done, run the test_accuracy.py file by replacing the states obtained and the relevant dataloader to obtain the final results.

