# PickApp
Package to perform apple picking experiments with a UR5 robotic manipulator and a 3-finger underactuated gripper 

## Contact
Alejandro Velasquez  
velasale@oregonstate.edu  
PhD Student in Robotics

## Description
This package takes datasets of apple picks performed with a robotic manipulator UR5e and a robotic gripper as end effector.
The picks are performed in a real tree and in an apple proxy that emulates the physics of the apple tree.
Hence, this package is useful for the user to:
i) Compare time-series plots from real and proxy picks
ii) Obtain statistics about the experiments, such as the distribution of cartesian and angular noise
iii) Do basic machine learning to check if - with the given data - it is possible to train a classifier and predict the outcome of the pick (e.g. successful, failed)


## Examples
### module: compare_picks.py

The following example runs a Random Forest Classifier (RFC), with 10 experiments to account for the classifier's stochasticity, with a depth of 5 branches, and utilizes 5 features
```
python machine_learning.py --experiments 10 --depth 5 --feature 5 --classifier rfc 
```


module: machine_learning.py


module: metadata_stats.py


## Installation



## Tips
Tips to write better functions  
https://pybit.es/articles/writing-better-functions/

Git cheat-sheet  
https://education.github.com/git-cheat-sheet-education.pdf

MarkDown cheat-sheet  
https://www.markdownguide.org/cheat-sheet/
