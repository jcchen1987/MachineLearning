/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_CROSSVALIDATION_H__
#define __ML_CROSSVALIDATION_H__
#include "ml_Learner.h"
#include "ml_DataSet.h"

#define K_FOLDER         1
#define LEAVE_ONE_OUT    2

class ML_EXPORT CCrossValidation
{
public:
    CCrossValidation(int nValidationType = K_FOLDER, int nK = 10);
    ~CCrossValidation();

    int Validate(CLearner *pLearner, CDataSet *pDataSet, double &dAvgError);
private:
    void RandPerm(vector<int> &v);
    void TestC(CLearner *pLearner, CDataSet *pDataSet, vector<int> &vnTestIdx, double &dError);
    void TestR(CLearner *pLearner, CDataSet *pDataSet, vector<int> &vnTestIdx, double &dError);
private:
    int m_eLearnTarget;
    int m_nK;
};
#endif // !__ML_CROSSVALIDATION_H__

