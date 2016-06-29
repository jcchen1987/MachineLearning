/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_ADABOOST_H__
#define __ML_ADABOOST_H__

#include "ml_Learner.h"
class CCart;
class CAdaBoost : public CLearner
{
public:
    CAdaBoost();
    ~CAdaBoost();
    virtual int SetParam(void *pvParam);
    virtual int Train(CDataSet *pDataSet, int *pnSampleIdx = NULL, int nSampleCnt = 0, int *pnFeatureIdx = NULL, int nFeatureCnt = 0);
    virtual int Predict(TypeF *pFeature, int &nLabel, double &dProb);
    virtual int Predict(TypeF *pFeature, TypeR *pResponse);
    virtual int Save(FILE *&pf);
    virtual int Load(FILE *&pf);
    virtual int GetLearnTargetType(ELearnTarget &eTarget);

private:
    void InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt);
    double GetScore(TypeF *pFeature);
    void UpdateWeight(CDataSet *pDataSet, vector<int> &vnSampleIdx, vector<double> &vdScore);
    void FindTheta(CDataSet *pDataSet, vector<int> &vnSampleIdx);

private:
    int m_nMaxTreeCnt;
    int m_nMaxTreeDepth;
    bool m_bRestartTrain;
    bool m_bSoftCascade;
    double m_dRecall;
    EBoostType m_eBoostType;
    vector<CCart> m_vCart;
    vector<double> m_vdAlpha;
    vector<double> m_vdTheta;
};
#endif // !__ML_ADABOOST_H__

