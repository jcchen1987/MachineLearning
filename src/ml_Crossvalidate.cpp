#include "ml_Crossvalidate.h"
#include <stdlib.h>
#include <string.h>
CCrossValidation::CCrossValidation(int nValidationType, int nK)
{
    m_eLearnTarget = nValidationType;
    m_nK = nK;
}

CCrossValidation::~CCrossValidation()
{
}

int CCrossValidation::Validate(CLearner * pLearner, CDataSet * pDataSet, double &dAvgError)
{
    int i = 0, j = 0, nRet = 0;
    if (NULL == pLearner || NULL == pDataSet) return -1;
    int nSampleCnt = pDataSet->GetSampleCnt();
    if (nSampleCnt < 2) return -1;
    if (m_eLearnTarget == LEAVE_ONE_OUT) m_nK = nSampleCnt;
    if (m_nK <= 0) return -1;

    vector<int> vnSampleIdx(nSampleCnt);
    for (i = 0; i < nSampleCnt; i++)
    {
        vnSampleIdx[i] = i;
    }
    RandPerm(vnSampleIdx);

    int nSampleCntPerFolder = nSampleCnt / m_nK;
    int nStartId = 0;    
    vector<int> vnTrainIdx(nSampleCnt - nSampleCntPerFolder);
    vector<int> vnTestIdx(nSampleCntPerFolder);

    double dSumError = 0;
    i = 0;
    for (i = 0; i < m_nK; i++, nStartId += nSampleCntPerFolder)
    {
        //prepare train & test sample index
        memcpy(&vnTrainIdx[0], &vnSampleIdx[0], nStartId * sizeof(int));
        memcpy(&vnTestIdx[0], &vnSampleIdx[nStartId], nSampleCntPerFolder * sizeof(int));
        if (nStartId + nSampleCntPerFolder < nSampleCnt)
            memcpy(&vnTrainIdx[nStartId], &vnSampleIdx[nStartId + nSampleCntPerFolder], (nSampleCnt - nSampleCntPerFolder - nStartId) * sizeof(int));
        //train
        nRet = pLearner->Train(pDataSet, &vnTrainIdx[0], vnTrainIdx.size());
        if (0 != nRet) return nRet;
        
        //test
        ELearnTarget eLearnerTarget;
        pLearner->GetLearnTargetType(eLearnerTarget);
        double dErr = 0;
        switch (eLearnerTarget)
        {
        case ELearnTarget_Classification:
            TestC(pLearner, pDataSet, vnTestIdx, dErr);
            //printf("===============Err = %.2lf%%======\n", dErr *100);
            break;
        case ELearnTarget_Regression:
            TestR(pLearner, pDataSet, vnTestIdx, dErr);
            break;
        }        
        dSumError += dErr;
    }
    dAvgError = dSumError / m_nK;
    return 0;
}

void CCrossValidation::RandPerm(vector<int>& v)
{
    int n = v.size();
    for (int i = 0; i < n; i++)
    {
        int j = rand() % n;
        int t = v[i];
        v[i] = v[j];
        v[j] = t;
    }
}

void CCrossValidation::TestC(CLearner * pLearner, CDataSet * pDataSet, vector<int>& vnTestIdx, double & dError)
{
    if (vnTestIdx.empty()) return;
    int r = 0;
    for (int j = 0; j < (int)vnTestIdx.size(); j++)
    {
        int nLabel;
        double dProb;
        int nSampleId = vnTestIdx[j];
        pLearner->Predict(pDataSet->GetFeature(nSampleId), nLabel, dProb);
        if (nLabel != pDataSet->GetLabel(nSampleId))
            r++;
    }
    dError = r * 1.0 / vnTestIdx.size();
}

void CCrossValidation::TestR(CLearner * pLearner, CDataSet * pDataSet, vector<int>& vnTestIdx, double & dError)
{
    if (vnTestIdx.empty()) return;
    int nLen = pDataSet->GetResponseDim();
    vector<TypeR> vRes(nLen);
    dError = 0;
    for (int i = 0; i < (int)vnTestIdx.size(); i++)
    {
        int nSampleId = vnTestIdx[i];
        pLearner->Predict(pDataSet->GetFeature(nSampleId), &vRes[0]);
        TypeR *pReal = pDataSet->GetResponse(nSampleId);

        for (int j = 0; j < nLen; j++)
        {
            dError += (vRes[j] - pReal[j]) * (vRes[j] - pReal[j]);
        }
    }
    dError /= vnTestIdx.size();
}
