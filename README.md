# MachineLearning

---
## Desciption

  This repository provided some basic machine learning algothms, but you can combine them to be a complex learner
in your own way.    
  All the learning algorithms are implemented with a uniform interface.
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

you can referance the main.cpp to find how to call each learner.

## How to complie this project
This project is organized with cmake.
you can use cmake-gui to generate the project on windows / linux
while in command line mode, you can generate and complie the project with the following commands:
> ***mkdir build***    
> ***cd build***    
> ***cmake ..***   
> ***make***    

Then use
> ***./bin/ml_test*** 

to run the program


## Others
Also, you can contact jcchen1987@163.com if you have any questions or advices about the reposity.
