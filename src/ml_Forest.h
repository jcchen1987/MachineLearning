/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_FOREST_H__
#define __ML_FOREST_H__

#include "ml_Learner.h"
class CCart;

class CForest : public CLearner
{
public:
    CForest();
    ~CForest();
    virtual int SetParam(void *pvParam);
    virtual int Train(CDataSet *pDataSet, int *pnSampleIdx = NULL, int nSampleCnt = 0, int *pnFeatureIdx = NULL, int nFeatureCnt = 0);
    virtual int Predict(TypeF *pFeature, int &nLabel, double &dProb);
    virtual int Predict(TypeF *pFeature, TypeR *pResponse);
    virtual int Save(FILE *&pf);
    virtual int Load(FILE *&pf);
    virtual int GetLearnTargetType(ELearnTarget &eTarget);
private:
    void InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt);
    void RandSelectWithRelacement(int *pnSrc, int nSrcCnt, int *pnDst, int nDstCnt);
    void RandSelectWithoutRelacement(int *pnSrc, int nSrcCnt, int *pnDst, int nDstCnt);
private:
    int m_nMaxTreeCnt;
    int m_nTreeDepth;
    ELearnTarget m_eLearnTarget;
    vector<CCart> m_vCart;
    double m_dSelectSampleRate;
    double m_dSelectFeatureRate;
    bool m_bRestartTrain;

    int m_nResponseLen;
};
#endif // !__ML_FOREST_H__

