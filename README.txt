Google Drive URL (anyone with the link can view)

https://drive.google.com/drive/folders/1RA4s6Ww9_sUz8vq7HgdJ8taWwLS-xC6u?usp=sharing

How to reproduce the plots in the npark62-analysis.pdf

1. Store all the .py codes and .csv files from the URL above in the same directory as CPeaks.py, flipflop.py, knapsack.py and NeuralNetWeighting.py (Ensure this exact mlrose package is used since it is modified to perform certain other functions.)

2. Run it in the directory, e.g. python knapsack.py

CPeaks.py, flipflop.py, knapsack.py include the graphs of all of the first part of the analysis where the simple optimization problems are created to be solved using the four optimization algorithms. This includes the graphs related to the hyperparameter tuning for all randomized optimization algorithms as well as the ones for algorithm comparison.

NeuralNetWeighting.py include the graphs of all of the second part of the analysis where the mushroom.csv is preprocessed and supervise-learned by the neural network whose weights are calculated by the randomized optimization algorithms.