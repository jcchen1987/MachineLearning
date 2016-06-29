#include "ml_AdaBoost.h"
#include "ml_Cart.h"
#include <algorithm>
#include <math.h>
#include <string.h>
#include <float.h>
#define LABEL_POS 1
#define LABEL_NEG -1

static double ProbMapDiscrete(int nLabel, double dProb)
{
    if (LABEL_NEG == nLabel) return -0.5;
    return 0.5;
}

static double ProbMapReal(int nLabel, double dProb)
{
    const double dEps = 0.01;
    double dPosProb = dProb;
    if (LABEL_NEG == nLabel) dPosProb = 1 - dProb;
    double dResProb = 0.5f * log((dPosProb + dEps) / (1 - dPosProb + dEps));
    return dResProb;
}

static double ProbMapGentle(int nLabel, double dProb)
{
    double dPosProb = dProb;
    if (LABEL_NEG == nLabel) dPosProb = 1 - dProb;
    return dPosProb - (1 - dPosProb);
}

CAdaBoost::CAdaBoost()
{
}

CAdaBoost::~CAdaBoost()
{
}

int CAdaBoost::SetParam(void * pvParam)
{
    if (NULL == pvParam) return -1;
    TAdaBoostParam *ptParam = (TAdaBoostParam *)pvParam;
    m_nMaxTreeCnt = ptParam->nMaxTreeCnt;
    m_nMaxTreeDepth = ptParam->nMaxTreeDepth;
    m_bRestartTrain = ptParam->bRestartTrain;
    m_bSoftCascade = ptParam->bSoftCascade;
    m_dRecall = ptParam->dRecall;
    m_eBoostType = ptParam->eBoostType;
    return 0;
}

int CAdaBoost::Train(CDataSet *pDataSet, int *pnSampleIdx, int nSampleCnt, int *pnFeatureIdx, int nFeatureCnt)
{
    int i, nRet;
    if (!pDataSet) return -1;
    int nSCnt = 0;
    int nFCnt = 0;
    nSCnt = pDataSet->GetSampleCnt();
    nFCnt = pDataSet->GetFeatureDim();
    if (nSCnt < 2 || nFCnt < 1) return -1;
    
    if (m_bRestartTrain)
    {
        m_vCart.clear();
        m_vdAlpha.clear();
        m_vdTheta.clear();
    }
    m_vCart.reserve(m_nMaxTreeCnt);
    m_vdAlpha.reserve(m_nMaxTreeCnt);
    m_vdTheta.reserve(m_nMaxTreeCnt);

    TCartParam tCartParam;
    tCartParam.eLearnTarget = ELearnTarget_Classification;
    tCartParam.eSplitRule = ESplitRule_MINERRWEIGHT;
    tCartParam.nMaxTreeDepth = m_nMaxTreeDepth;
    switch (m_eBoostType)
    {
    case EBoostType_Discrete: tCartParam.pfProbMapFun = ProbMapDiscrete; break;
    case EBoostType_Real:     tCartParam.pfProbMapFun = ProbMapReal; break;
    case EBoostType_Gentle:   tCartParam.pfProbMapFun = ProbMapGentle; break;        
    }
    
    //inital sample and feature index if it is empty
    vector<int> vnSampleIdx;
    vector<int> vnFeatureIdx;
    InitIndex(vnSampleIdx, vnFeatureIdx, nSCnt, nFCnt, pnSampleIdx, nSampleCnt, pnFeatureIdx, nFeatureCnt);
    pnSampleIdx = &vnSampleIdx[0];
    nSampleCnt = vnSampleIdx.size();
    pnFeatureIdx = &vnFeatureIdx[0];
    nFeatureCnt = vnFeatureIdx.size();

    //inital score & pos/neg sample count
    //backup sample weight, because it may be changed during the learning process
    vector<double> vdWeightBackup(nSampleCnt);
    vector<double> vdScore(nSampleCnt);
    int nPosCnt = 0;
    int nNegCnt = 0;
    for (i = 0; i < nSampleCnt; i++)
    {
        int nSampleId = pnSampleIdx[i];
        int nLabel = pDataSet->GetLabel(nSampleId);
        switch (nLabel)
        {
        case LABEL_POS: nPosCnt++; break;            
        case LABEL_NEG: nNegCnt++; break;
        default: return -1; break;
        }
        vdScore[i] = GetScore(pDataSet->GetFeature(nSampleId));
        vdWeightBackup[i] = pDataSet->GetWeight(nSampleId);
        pDataSet->SetWeight(nSampleId, 1); 
    }
    for (i = 0; i < nSampleCnt; i++)
    {
        int nSampleId = pnSampleIdx[i];
        int nLabel = pDataSet->GetLabel(nSampleId);
        switch (nLabel)
        {
        case LABEL_POS: pDataSet->SetWeight(nSampleId, nNegCnt); break;
        case LABEL_NEG: pDataSet->SetWeight(nSampleId, nPosCnt); break;
        default: return -1; break;
        }
    }
    
    while ((int)m_vCart.size() < m_nMaxTreeCnt)
    {
        //update weight
        UpdateWeight(pDataSet, vnSampleIdx, vdScore);

        //train cart
        m_vCart.push_back(CCart());
        CCart &CurCart = m_vCart.at(m_vCart.size() - 1);
        nRet = CurCart.SetParam(&tCartParam);
        if (nRet != 0) return nRet;
        nRet = CurCart.Train(pDataSet, pnSampleIdx, nSampleCnt, pnFeatureIdx, nFeatureCnt);
        if (nRet != 0) return nRet;

        //predict with current tree
        int nLabel = 0;
        double dErrWeight = 0;
        for (i = 0; i < nSampleCnt; i++)
        {
            nRet = CurCart.Predict(pDataSet->GetFeature(pnSampleIdx[i]), nLabel, vdScore[i]);
            if (nRet != 0) return nRet; 
            nLabel = pDataSet->GetLabel(pnSampleIdx[i]);
            if (nLabel * vdScore[i] < 0) dErrWeight += pDataSet->GetWeight(pnSampleIdx[i]);
        }
        
        //calc alpha
        double dAlpha = 1;
        if (m_eBoostType == EBoostType_Discrete)
        {
            dAlpha = log((1 - dErrWeight) / (dErrWeight + 1e-20));            
            //printf("alpha=%lf  err=%.2lf%%\n", dAlpha, dErrWeight*100);
        }       
        m_vdAlpha.push_back(dAlpha);

        for (i = 0; i < nSampleCnt; i++)
        {
            vdScore[i] *= dAlpha;
        }
    }

    //softcascade todo.
    FindTheta(pDataSet, vnSampleIdx);
    
    //restore sample weight
    for (i = 0; i < nSampleCnt; i++)
    {
        pDataSet->SetWeight(pnSampleIdx[i], vdWeightBackup[i]);
    }
    return 0;
}

int CAdaBoost::Predict(TypeF * pFeature, int & nLabel, double & dProb)
{
    if (m_vCart.empty()) return -1;
    double dScore = 0;
    double dCurScore = 0;
    int nCurLable = 0;
    nLabel = LABEL_POS;

    for (int i = 0; i < (int)m_vCart.size(); i++)
    {
        m_vCart[i].Predict(pFeature, nCurLable, dCurScore);
        dScore += m_vdAlpha[i] * dCurScore;
        if (dScore < m_vdTheta[i])
        {
            nLabel = LABEL_NEG;
            break;
        }
    }
    //printf("%lf\n", dScore);
    //the prob can be calculated by P(nLabel = LABEL_POS) = exp(s)/(exp(s) + exp(-s)) where s is the score
    //but it is always not neccessary and may lead to high latency in some cases, such as face/pedestrian detection
    dProb = 1;
    return 0;
}

int CAdaBoost::Predict(TypeF * pFeature, TypeR * pResponse)
{
    return -1;
}

int CAdaBoost::Save(FILE *& pf)
{
    if (NULL == pf) return -1;
    fwrite(&m_nMaxTreeCnt, sizeof(int), 1, pf);
    fwrite(&m_nMaxTreeDepth, sizeof(int), 1, pf);
    fwrite(&m_bRestartTrain, sizeof(bool), 1, pf);
    fwrite(&m_bSoftCascade, sizeof(bool), 1, pf);
    fwrite(&m_dRecall, sizeof(double), 1, pf);
    fwrite(&m_eBoostType, sizeof(EBoostType), 1, pf);
  
    int nLen;
    nLen = m_vCart.size();
    fwrite(&nLen, sizeof(int), 1, pf);
    if (nLen > 0)
    {
        fwrite(&m_vdAlpha[0], sizeof(double), nLen, pf);
        fwrite(&m_vdTheta[0], sizeof(double), nLen, pf);
    }
    
    for (int i = 0; i < nLen; i++)
    {
        m_vCart[i].Save(pf);
    }
    return 0;
}

int CAdaBoost::Load(FILE *& pf)
{
    if (NULL == pf) return -1;
    fread(&m_nMaxTreeCnt, sizeof(int), 1, pf);
    fread(&m_nMaxTreeDepth, sizeof(int), 1, pf);
    fread(&m_bRestartTrain, sizeof(bool), 1, pf);
    fread(&m_bSoftCascade, sizeof(bool), 1, pf);
    fread(&m_dRecall, sizeof(double), 1, pf);
    fread(&m_eBoostType, sizeof(EBoostType), 1, pf);

    int nLen;
    fread(&nLen, sizeof(int), 1, pf);
    if (nLen > 0)
    {
        m_vCart.resize(nLen);
        m_vdAlpha.resize(nLen);
        m_vdTheta.resize(nLen);
        fwrite(&m_vdAlpha[0], sizeof(double), nLen, pf);
        fwrite(&m_vdTheta[0], sizeof(double), nLen, pf);
    }
    for (int i = 0; i < nLen; i++)
    {
        m_vCart[i].Save(pf);
    }
    return 0;
}

int CAdaBoost::GetLearnTargetType(ELearnTarget & eTarget)
{
    eTarget = ELearnTarget_Classification;
    return 0;
}

void CAdaBoost::InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt)
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

double CAdaBoost::GetScore(TypeF * pFeature)
{
    double dScore = 0;
    double dCurScore = 0;
    int nLabel = 0;
    for (int i = 0; i < (int)m_vCart.size(); i++)
    {
        m_vCart[i].Predict(pFeature, nLabel, dCurScore);
        dScore += m_vdAlpha[i] * dCurScore;
    }
    return dScore;
}

void CAdaBoost::UpdateWeight(CDataSet *pDataSet, vector<int> &vnSampleIdx, vector<double> &vdScore)
{
    double dSumWeight = 0;
    vector<double> vdWeightTmp(vnSampleIdx.size());

    for (int i = 0; i < (int)vnSampleIdx.size(); i++)
    {
        int nSampleId = vnSampleIdx[i];
        int nLabel = pDataSet->GetLabel(nSampleId);
        double dCurW = pDataSet->GetWeight(nSampleId);
        vdWeightTmp[i] = dCurW * exp(-nLabel * vdScore[i]);
        dSumWeight += vdWeightTmp[i];
    }

    dSumWeight = 1.0 / dSumWeight;
    for (int i = 0; i < (int)vnSampleIdx.size(); i++)
    {
        int nSampleId = vnSampleIdx[i];
        double w = vdWeightTmp[i] * dSumWeight;
        pDataSet->SetWeight(nSampleId, w);
    }
}

bool FinalScoreDesent(vector<double> v1, vector<double> v2)
{
    if (v1[v1.size() - 1] > v2[v2.size() - 1]) return true;
    return false;

}
void CAdaBoost::FindTheta(CDataSet * pDataSet, vector<int>& vnSampleIdx)
{
    vector< vector<double> > vvdScore;
    int nScoreCnt = m_vCart.size();
    for (int i = 0; i < (int)vnSampleIdx.size(); i++)
    {
        if (LABEL_POS != pDataSet->GetLabel(vnSampleIdx[i])) continue;
        TypeF *pFeature = pDataSet->GetFeature(vnSampleIdx[i]);
        double dScore = 0;
        double dCurScore = 0;
        int nCurLable = 0;
        vvdScore.push_back(vector<double>());
        vector<double> &vCurSampleScore = vvdScore.at(vvdScore.size() - 1);
        vCurSampleScore.resize(nScoreCnt);
        for (int i = 0; i < nScoreCnt; i++)
        {
            m_vCart[i].Predict(pFeature, nCurLable, dCurScore);
            dScore += m_vdAlpha[i] * dCurScore;
            vCurSampleScore[i] = dScore;
        }
    }
    std::sort(vvdScore.begin(), vvdScore.end(), FinalScoreDesent);
    int nThIdx = int(vvdScore.size() * m_dRecall + 0.5) ;
    
    m_vdTheta.resize(nScoreCnt, -DBL_MAX);
    m_vdTheta[nScoreCnt - 1] = vvdScore[nThIdx - 1].at(nScoreCnt - 1);
    if (!m_bSoftCascade) return;
    for (int i = nScoreCnt - 2; i >= 0; i--)
    {
        m_vdTheta[i] = vvdScore[0].at(i);
        for (int j = 0; j < nThIdx; j++)
        {
            if (m_vdTheta[i] > vvdScore[j].at(i)) m_vdTheta[i] = vvdScore[j].at(i);
        }
    }
}
