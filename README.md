# PickApp
Package to perform apple picking experiments with a UR5 robotic manipulator and a 3-finger underactuated gripper. 

## Contact
Alejandro Velasquez  
velasale@oregonstate.edu  
Robotics PhD Student

## Description
This package takes datasets of apple picks performed with a robotic manipulator UR5e and a robotic gripper as end effector.
The picks are performed in a real tree and in an apple proxy that emulates the physics of the apple tree.
Hence, this package is useful for the user to:
- [x] Compare time-series plots from real and proxy picks.
- [x] Obtain statistics about the experiments, such as the distribution of cartesian and angular noise.
- [x] Do basic machine learning to check if - with the given data - it is possible to train a classifier and predict the outcome of the pick (e.g. successful, failed).


## Examples
### module: compare_picks.py
The following example analyzes channel 'Force_x', among the 'failed' picks, and does the Dynamic Time Warping (DTW) analysis during the 'pick' phase.
```
python compare_picks.py --variable force_x --case failed --phase pick
```


### module: machine_learning.py
The following example runs a Random Forest Classifier (RFC), with 10 experiments to account for the classifier's stochasticity, with a depth of 5 branches, and utilizes 5 features.
```
python machine_learning.py --experiments 10 --depth 5 --feature 5 --classifier rfc 
```
It outputs a boxplot with the classifier's accuracies during the experiments.
The boxplot gets stored in a .pdf file, along with a .txt file with the confussion matrix of the best accuracy.


### module: metadata_stats.py
In the following example, the statistic analysis is run for the dataset '3_proxy_winter22_x1'.
```
python metadata_stats.py --dataset 3_proxy_winter22_x1
```
It outputs .pdfs with boxplots of the angular and cartesian noise.
It also outputs a .txt file with Mean, SD and percentiles of each noise.
These files are stored in 'results/'



## Installation



## Tips
Tips to write better functions  
https://pybit.es/articles/writing-better-functions/

Git cheat-sheet  
https://education.github.com/git-cheat-sheet-education.pdf

MarkDown cheat-sheet  
https://www.markdownguide.org/cheat-sheet/
