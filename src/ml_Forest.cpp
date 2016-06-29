#include "ml_Forest.h"
#include "ml_Cart.h"
#include <stdlib.h>
#include <string.h>

CForest::CForest()
{
    m_nMaxTreeCnt = 10;
    m_nTreeDepth = 4;
    m_eLearnTarget = ELearnTarget_Classification;
    m_vCart.clear();
    m_dSelectSampleRate = 1;
    m_dSelectFeatureRate = 0.8;
    m_bRestartTrain = true;
    m_nResponseLen = 0;
}

CForest::~CForest()
{
}

int CForest::SetParam(void * pvParam)
{
    TForestParam *ptParam = (TForestParam *)pvParam;
    if (NULL == ptParam) return -1;
    m_nMaxTreeCnt = ptParam->nMaxTreeCnt;
    m_nTreeDepth = ptParam->nMaxTreeDepth;
    m_eLearnTarget = ptParam->eLearnTarget;
    m_dSelectSampleRate = ptParam->dSelectSampleRate;
    m_dSelectFeatureRate = ptParam->dSelectFeatureRate;

    m_bRestartTrain = ptParam->bRestartTrain;
    return 0;
}

int CForest::Train(CDataSet *pDataSet, int *pnSampleIdx, int nSampleCnt, int *pnFeatureIdx, int nFeatureCnt)
{
    int nRet;
    if (!pDataSet) return -1;
    int nSCnt = 0;
    int nFCnt = 0;
    nSCnt = pDataSet->GetSampleCnt();
    nFCnt = pDataSet->GetFeatureDim();
    if (nSCnt < 2 || nFCnt < 1) return -1;
    
    m_nResponseLen = pDataSet->GetResponseDim();
    if (m_eLearnTarget == ELearnTarget_Regression && m_nResponseLen <= 0) return -1;
    
    if (m_bRestartTrain)
    {
        m_vCart.clear();        
    }
    m_vCart.reserve(m_nMaxTreeCnt);

    //inital sample and feature index if it is empty
    vector<int> vnSampleIdx;
    vector<int> vnFeatureIdx;
    InitIndex(vnSampleIdx, vnFeatureIdx, nSCnt, nFCnt, pnSampleIdx, nSampleCnt, pnFeatureIdx, nFeatureCnt); 
    nSampleCnt = vnSampleIdx.size();
    nFeatureCnt = vnFeatureIdx.size();

    int nSubSampleCnt = (int)(m_dSelectSampleRate * nSampleCnt + 0.5);
    int nSubFeatureCnt = (int)(m_dSelectFeatureRate * nFeatureCnt + 0.5);
    vector<int> vnSubSampleIdx(nSubSampleCnt);
    vector<int> vnSubFeatureIdx(nSubFeatureCnt);
    TCartParam tCartParam;
    tCartParam.eLearnTarget = m_eLearnTarget;
    tCartParam.eSplitRule = ESplitRule_GINI;
    tCartParam.nMaxTreeDepth = m_nTreeDepth;
    tCartParam.pfProbMapFun = NULL;
    while((int)m_vCart.size() < m_nMaxTreeCnt)    
    {
        m_vCart.push_back(CCart());
        CCart &CurCart = m_vCart.at(m_vCart.size() - 1);
        nRet = CurCart.SetParam(&tCartParam);
        if (nRet != 0) return nRet;

        //select sample idx
        RandSelectWithRelacement(&vnSampleIdx[0], vnSampleIdx.size(), &vnSubSampleIdx[0], vnSubSampleIdx.size());

        //select feature idx
        RandSelectWithoutRelacement(&vnFeatureIdx[0], vnFeatureIdx.size(), &vnSubFeatureIdx[0], vnSubFeatureIdx.size());
        nRet = CurCart.Train(pDataSet, &vnSubSampleIdx[0], nSubSampleCnt, &vnSubFeatureIdx[0], nSubFeatureCnt);
        if (nRet != 0) return nRet;            
    }
    return nRet;
}

int CForest::Predict(TypeF * pFeature, int & nLabel, double & dProb)
{
    int nRet = 0;
    map<int, int> mLableCount;
    map<int, int>::iterator it;
    int nCurLabel = 0;
    double dCurProb = 0;
    
    if (m_vCart.empty()) return -1;

    for (int i = 0; i < (int)m_vCart.size(); i++)
    {
        nRet = m_vCart[i].Predict(pFeature, nCurLabel, dCurProb);
        if (nRet != 0) return nRet;

        it = mLableCount.find(nCurLabel);
        if (it == mLableCount.end())
        {
            mLableCount.insert(std::make_pair(nCurLabel, 1));
        }
        else
        {
            it->second++;
        }
    }

    //vote
    it = mLableCount.begin();
    nLabel = 0;
    dProb = 0;    
    do
    {
        if (dProb < it->second)
        {
            nLabel = it->first;
            dProb = it->second;
        }
        it++;
    } while (it != mLableCount.end());
    dProb = dProb * 1.0 / (int)m_vCart.size();

    return 0;
}

int CForest::Predict(TypeF *pFeature, TypeR *pResponse)
{
    int nRet = 0;
    if (m_vCart.empty() || m_nResponseLen <= 0) return -1;

    vector<TypeR> vSumResponse(m_nResponseLen, 0);
    vector<TypeR> vResponse(m_nResponseLen, 0);
    for (int i = 0; i < (int)m_vCart.size(); i++)
    {
        nRet = m_vCart[i].Predict(pFeature, &vResponse[0]);
        if (nRet != 0) return nRet;
        for (int j = 0; j < m_nResponseLen; j++)
        {
            vSumResponse[j] += vResponse[j];
        }
    }
    double dCof = 1.0 / m_vCart.size();
    for (int j = 0; j < m_nResponseLen; j++)
    {
        pResponse[j] = vSumResponse[j] * dCof;
    }
    return 0;
}

int CForest::Save(FILE *& pf)
{
    if (NULL == pf) return -1;
    fwrite(&m_nMaxTreeCnt, sizeof(int), 1, pf);
    fwrite(&m_nTreeDepth, sizeof(int), 1, pf);
    fwrite(&m_eLearnTarget, sizeof(ELearnTarget), 1, pf);
    fwrite(&m_dSelectSampleRate, sizeof(double), 1, pf);
    fwrite(&m_dSelectFeatureRate, sizeof(double), 1, pf);
    fwrite(&m_bRestartTrain, sizeof(bool), 1, pf);

    int nLen = m_vCart.size();
    fwrite(&nLen, sizeof(int), 1, pf);
    for (int i = 0; i < nLen; i++)
    {
        m_vCart[i].Save(pf);
    }
    return 0;
}

int CForest::Load(FILE *& pf)
{
    if (NULL == pf) return -1;
    fread(&m_nMaxTreeCnt, sizeof(int), 1, pf);
    fread(&m_nTreeDepth, sizeof(int), 1, pf);
    fread(&m_eLearnTarget, sizeof(ELearnTarget), 1, pf);
    fread(&m_dSelectSampleRate, sizeof(double), 1, pf);
    fread(&m_dSelectFeatureRate, sizeof(double), 1, pf);
    fread(&m_bRestartTrain, sizeof(bool), 1, pf);

    int nLen;
    fread(&nLen, sizeof(int), 1, pf);
    m_vCart.resize(nLen);
    for (int i = 0; i < nLen; i++)
    {
        m_vCart[i].Save(pf);
    }
    return 0;
}

int CForest::GetLearnTargetType(ELearnTarget & eTarget)
{
    eTarget = m_eLearnTarget;
    return 0;
}

void CForest::InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt)
{
    int i;
    if (NULL == pnSampleIdx || nSampleCnt <= 1)
    {
        nSampleCnt = nAllSmpCnt;
        vnSampleIdx.resize(nSampleCnt);
        for (i = 0; i < nSampleCnt; i++)  vnSampleIdx[i] = i;
    }
    else
    {
        vnSampleIdx.resize(nSampleCnt);
        memcpy(&vnSampleIdx[0], pnSampleIdx, sizeof(int) * nSampleCnt);
    }

    if (NULL == pnFeatureIdx || nFeatureCnt <= 0)
    {
        nFeatureCnt = nAllFeaCnt;
        vnFeatureIdx.resize(nFeatureCnt);
        for (i = 0; i < nFeatureCnt; i++) vnFeatureIdx[i] = i;
    }
    else
    {
        vnFeatureIdx.resize(nFeatureCnt);
        memcpy(&vnFeatureIdx[0], pnFeatureIdx, sizeof(int) * nFeatureCnt);
    }
}

void CForest::RandSelectWithRelacement(int *pnSrc, int nSrcCnt, int *pnDst, int nDstCnt)
{
    for (int i = 0; i < nDstCnt; i++)
    {
        int nRandNum = rand() % nSrcCnt;
        pnDst[i] = pnSrc[nRandNum];
    }
}

void CForest::RandSelectWithoutRelacement(int *pnSrc, int nSrcCnt, int *pnDst, int nDstCnt)
{
    int i = 0;
    vector<bool> vbSelected(nSrcCnt, false);
    while (i < nDstCnt)
    {
        int nRandNum = rand() % nSrcCnt;
        if (vbSelected[nRandNum]) continue;
        vbSelected[nRandNum] = true;
        pnDst[i] = pnSrc[nRandNum];
        i++;
    }
}

