# MachineLearning

---
## Desciption

  This repository provided some basic machine learning algothms, which can be combined flexibly due to your own application. such as  bootstrap train in face detection, pedestrian detection or other else. And boosting learners(bagging, random forest, adaboost) are designed to run step by step if needed. For example, if you want to train a random forest with 1000 trees, you can train 200 trees firstly, then train 300 trees later,...,until the learner meets you demands. I think this may help you saving much time to analyize the optimal performence parameters. 
  
  All the learning algorithms are implemented with a set of uniform interface.
The main frame of the repository is as follows:    

    |
    |-----CLearner
    |         |-----CCart
    |         |-----CForest
    |         |-----CAdaboost
    |         |----- ...
    |         |----- ...
    |
    |-----CDataSet
    |         |-----you can implement a dataset class with inherit the CDataSet
    |               a class CMyDataSet has been presented in ./test/main.cpp
    |
    |-----CCrossValidate(this class implemented the k-folder and leave-one-out cross validation

you can get how to use each learner in main.cpp.

## How to complie this project
This project is organized with cmake.    
you can use ***cmake-gui*** to generate the project on windows / linux.    
while in command line mode, you can generate and complie the project with the following commands:
> ***mkdir build***    
> ***cd build***    
> ***cmake ..***   
> ***make***    

Then use
> ***./bin/ml_test*** 

to run the program


## Others
This code is for general usage. So it can be optimized acordding to your usage. You can do it by your self.

Also, you can contact jcchen1987@163.com if you have any questions or advices about the reposity.
