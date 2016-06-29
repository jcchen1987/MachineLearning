/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_LEARNER_H__
#define __ML_LEARNER_H__
#include "ml_DataSet.h"
#include <stdio.h>

enum ELearnTarget
{
    ELearnTarget_Classification = 1,
    ELearnTarget_Regression = 2,
};

//basic learner class
class ML_EXPORT CLearner
{
public:
    //set algorithm parameters, pvParam must be convered from the right struct due to the learner type
    //ELearner_Cart -- TCartParam
    //ELearner_Forest -- TForestParam
    //ELearner_AdaBoost -- TAdaBoostParam
    virtual int SetParam(void *pvParam) = 0;

    //learner train interface, pnSampleIdx/pnFeatureIdx means which samples are used to train learner
    //NULL means using all.
    virtual int Train(CDataSet *pDataSet, int *pnSampleIdx = NULL, int nSampleCnt = 0, int *pnFeatureIdx = NULL, int nFeatureCnt = 0) = 0;
    
    //you can call this interface to predict the regression problem
    virtual int Predict(TypeF *pFeature, TypeR *pResponse) = 0;

    //you can call this interface to predict the classification problem
    virtual int Predict(TypeF *pFeature, int &nLabel, double &dProb) = 0;

    //save or load learner model. 
    //when you calling Save, the pf must be opened with 'wb',
    //and 'rb' with calling Load
    virtual int Save(FILE *&pf) = 0;
    virtual int Load(FILE *&pf) = 0;

    virtual int GetLearnTargetType(ELearnTarget &eTarget) = 0;
};

enum ELearner
{
    ELearner_Cart = 0,
    ELearner_Forest = 1,
    ELearner_AdaBoost = 2,
};

ML_EXPORT CLearner *CreateLearner(ELearner eLearner);
ML_EXPORT void DestroyLearner(CLearner *pLearner);


//====================================================================================
// the follow structs are defined for the different learning algorithms to set optional parameters

//the function point which used to map the confidence of the tree leaf node
//it is always used in boosting
typedef double(*PFProbMap)(int nLabel, double dProb);

//this rule ESplitRule_MINERRWEIGHT is better for boosting
//but its performence  may be worse than the former 2 rules in a single tree
enum ETreeSplitRule
{
    ESplitRule_GINI = 1,
    ESplitRule_Entropy = 2,
    ESplitRule_MINERRWEIGHT = 3 
};

//there are only 3 type boost learning algorithms that have been implemented.
//others may be added by your self if you need.
enum EBoostType
{
    EBoostType_Discrete = 1,
    EBoostType_Real = 2,
    EBoostType_Gentle = 3
};

struct TCartParam
{
    int nMaxTreeDepth;
    ELearnTarget eLearnTarget;
    ETreeSplitRule eSplitRule;
    PFProbMap pfProbMapFun;
};

struct TForestParam
{
    int nMaxTreeCnt;
    int nMaxTreeDepth;
    ELearnTarget eLearnTarget;
    double dSelectSampleRate;
    double dSelectFeatureRate;
    bool bRestartTrain;
};

struct TAdaBoostParam
{
    int nMaxTreeCnt;
    int nMaxTreeDepth;
    EBoostType eBoostType;
    bool bRestartTrain;
    bool bSoftCascade;
    double dRecall;
};

#endif
