#include "ml_Cart.h"
#include <set>
#include <string.h>
#include <math.h>
#include <stdlib.h>
using std::set;

#define INFO_GINI 1
#define INFO_ENTROPY 2
#define INFO_MINERRWEIGHT 3

void TTreeNode::Save(FILE *&pf)
{
    fwrite(&m_nIdx, sizeof(int), 1, pf);
    fwrite(&m_FeatureTh, sizeof(TypeF), 1, pf);
    fwrite(&m_nLeftId, sizeof(int), 1, pf);
    fwrite(&m_nRightId, sizeof(int), 1, pf);

    int nLen = m_vResponse.size();
    fwrite(&nLen, sizeof(int), 1, pf);
    if(nLen > 0) fwrite(&m_vResponse[0], sizeof(TypeR), nLen, pf);

    fwrite(&m_nLabel, sizeof(int), 1, pf);
    fwrite(&m_dCorrectW, sizeof(int), 1, pf);
    fwrite(&m_dErrorW, sizeof(int), 1, pf);
    fwrite(&m_fConfidence, sizeof(double), 1, pf);
}
void TTreeNode::Load(FILE *&pf)
{
    fread(&m_nIdx, sizeof(int), 1, pf);
    fread(&m_FeatureTh, sizeof(TypeF), 1, pf);
    fread(&m_nLeftId, sizeof(int), 1, pf);
    fread(&m_nRightId, sizeof(int), 1, pf);

    int nLen;
    fread(&nLen, sizeof(int), 1, pf);
    if (nLen > 0)
    {
        m_vResponse.resize(nLen);
        fread(&m_vResponse[0], sizeof(TypeR), nLen, pf);
    }
    
    fread(&m_nLabel, sizeof(int), 1, pf);
    fread(&m_dCorrectW, sizeof(int), 1, pf);
    fread(&m_dErrorW, sizeof(int), 1, pf);
    fread(&m_fConfidence, sizeof(double), 1, pf);
}

//=====================================================
CCart::CCart()
{
    m_eLearnTarget = ELearnTarget_Classification;
    m_nMaxDepth = 2;
    m_nSplitRule = INFO_GINI;
    m_vtNode.clear();
    m_mLabelMap.clear();
    m_vLabelMapInv.clear();
    m_pfProbMap = NULL;
    m_nLeafNodeId = -1;
}

CCart::~CCart()
{
}

int CCart::SetParam(void * pvParam)
{
    if (NULL == pvParam) return -1;
    TCartParam *ptParam = (TCartParam *)pvParam;
    m_eLearnTarget = ptParam->eLearnTarget;
    m_nSplitRule = (int)ptParam->eSplitRule;
    m_pfProbMap = ptParam->pfProbMapFun;
    m_nMaxDepth = ptParam->nMaxTreeDepth;

    return 0;
}
int CCart::Train(CDataSet *pDataSet, int *pnSampleIdx, int nSampleCnt, int *pnFeatureIdx, int nFeatureCnt)
{
    int i = 0;
    if (!pDataSet) return -1;
    int nSCnt = 0;
    int nFCnt = 0;
    nSCnt = pDataSet->GetSampleCnt();
    nFCnt = pDataSet->GetFeatureDim();
    if (nSCnt < 2 || nFCnt < 1) return -1;
        
    //inital sample and feature index if it is empty 
    vector<int> vnSampleIdx;
    vector<int> vnFeatureIdx;
    InitIndex(vnSampleIdx, vnFeatureIdx, nSCnt, nFCnt, pnSampleIdx, nSampleCnt, pnFeatureIdx, nFeatureCnt);
    nSampleCnt = vnSampleIdx.size();
    nFeatureCnt = vnFeatureIdx.size();

    //map the class label to [0, c-1]
    if (m_eLearnTarget == ELearnTarget_Classification)
    {
        MapIdx(pDataSet, &vnSampleIdx[0], nSampleCnt);
    }

    //train cart
    m_vtNode.clear();
    m_vtNode.push_back(TTreeNode());
    SplitNode(0, pDataSet, nSampleCnt, nFeatureCnt, &vnSampleIdx[0], &vnFeatureIdx[0], 0);
    return 0;
}

int CCart::Predict(TypeF *pFeature, int &nLabel, double &dProb)
{
    if (m_eLearnTarget == ELearnTarget_Regression) return -1;
    m_nLeafNodeId = 0;
    if (!pFeature || m_vtNode.empty()) return -1;
    TTreeNode* pNode = FindLeafNode(pFeature, m_nLeafNodeId);

    nLabel = pNode->m_nLabel;
    dProb = pNode->m_fConfidence;
    return 0;
}

int CCart::Predict(TypeF * pFeature, TypeR * pResponse)
{
    if (m_eLearnTarget == ELearnTarget_Classification) return -1;
    m_nLeafNodeId = 0;
    if (!pFeature || m_vtNode.empty()) return -1;
    TTreeNode* pNode = FindLeafNode(pFeature, m_nLeafNodeId);
    memcpy(pResponse, &pNode->m_vResponse[0], pNode->m_vResponse.size() * sizeof(TypeR));
    return 0;
}

int CCart::Save(FILE *& pf)
{
    if (pf == NULL) return -1;
    fwrite(&m_eLearnTarget, sizeof(ELearnTarget), 1, pf);
    fwrite(&m_nMaxDepth, sizeof(int), 1, pf);
    fwrite(&m_nSplitRule, sizeof(int), 1, pf);

    int nLen = m_vtNode.size();
    fwrite(&nLen, sizeof(int), 1, pf);
    for (int i = 0; i < nLen; i++)
    {
        m_vtNode[i].Save(pf);
    }
    return 0;
}

int CCart::Load(FILE *& pf)
{
    if (pf == NULL) return -1;
    fread(&m_eLearnTarget, sizeof(ELearnTarget), 1, pf);
    fread(&m_nMaxDepth, sizeof(int), 1, pf);
    fread(&m_nSplitRule, sizeof(int), 1, pf);

    int nLen = 0;
    fread(&nLen, sizeof(int), 1, pf);
    m_vtNode.resize(nLen);
    for (int i = 0; i < nLen; i++)
    {
        m_vtNode[i].Load(pf);
    }
    return 0;
}

int CCart::GetLearnTargetType(ELearnTarget &eTarget)
{
    eTarget = m_eLearnTarget;
    return 0;
}

void CCart::MapIdx(CDataSet * ptTrainData, int * pnSampleIdx, int nSampleCnt)
{
    set<int> sLabel;
    int nLabel;
    for (int i = 0; i < nSampleCnt; i++)
    {
        int nSampleId = pnSampleIdx[i];
        nLabel = ptTrainData->GetLabel(nSampleId);
        sLabel.insert(nLabel);
    }
    set<int>::iterator it = sLabel.begin();
    m_mLabelMap.clear();
    m_vLabelMapInv.clear();
    m_vLabelMapInv.reserve(sLabel.size());

    int c = 0;
    while (it != sLabel.end())
    {
        m_mLabelMap.insert(std::make_pair(*it, c));
        m_vLabelMapInv.push_back(*it);
        it++;
        c++;
    }
}

void CCart::InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt)
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

void CCart::SplitNode(int nFatherNodeId, CDataSet *ptTrainData, int nSampleCnt, int nFeatureCnt,
    int *pnSampleIdx, int *pnFeatureIdx, int nCurDepth)
{
    int i = 0;
    TTreeNode *ptCurNode = &m_vtNode[nFatherNodeId];
    
    //check if it should end split
    bool bEndSplit = (nCurDepth == m_nMaxDepth);
    if (!bEndSplit)
    {
        switch (m_eLearnTarget)
        {
        case ELearnTarget_Classification:
            bEndSplit = IsNodePureC(ptTrainData, nSampleCnt, pnSampleIdx);
            break;
        case ELearnTarget_Regression:
            bEndSplit = IsNodePureR(ptTrainData, nSampleCnt, pnSampleIdx);
            break;
        }
    }

    //find best cut of the feature dim & threshold id, m_pnSampleIdx has been changed. <=threshold id is left and other is right
    int nThDim = -1;
    int nThIdx = 0;
    int nLeftcnt = 0;
    int nRightCnt = 0;
    if (!bEndSplit)
    {
        BestCut(ptTrainData, nSampleCnt, nFeatureCnt, pnSampleIdx, pnFeatureIdx, nThDim, nThIdx);
        nLeftcnt = nThIdx + 1;
        nRightCnt = nSampleCnt - nThIdx - 1;
        if (nLeftcnt == 0 || nRightCnt == 0)
        {
            bEndSplit = true;
        }
    }

    //stop split, generate the leaf node
    if (bEndSplit)
    {
        if(m_eLearnTarget == ELearnTarget_Classification)
            GenLeafNodeC(ptCurNode, ptTrainData, nSampleCnt, pnSampleIdx);
        else
            GenLeafNodeR(ptCurNode, ptTrainData, nSampleCnt, pnSampleIdx);
        return;
    } 

    //recursive split--nThIdx is between 0 and nSampleCnt-2, so nThDim + 1 will always be available
    ptCurNode->m_nIdx = nThDim;
    int nThSampleId0 = pnSampleIdx[nThIdx];
    int nThSampleId1 = pnSampleIdx[nThIdx + 1];

    TypeF *pdFeature0 = ptTrainData->GetFeature(nThSampleId0);
    TypeF *pdFeature1 = ptTrainData->GetFeature(nThSampleId1);
    ptCurNode->m_FeatureTh = (pdFeature0[nThDim] + pdFeature1[nThDim]) / 2;

 
    m_vtNode.push_back(TTreeNode());
    //it must use m_vtNode[nFatherNodeId] cause the ptCurNode maybe changed after push_back, the following is as the same
    m_vtNode[nFatherNodeId].m_nLeftId = m_vtNode.size() - 1;
    m_vtNode.push_back(TTreeNode());
    m_vtNode[nFatherNodeId].m_nRightId = m_vtNode.size() - 1;

    SplitNode(m_vtNode[nFatherNodeId].m_nLeftId, ptTrainData, nLeftcnt, nFeatureCnt, pnSampleIdx, pnFeatureIdx, nCurDepth + 1);
    SplitNode(m_vtNode[nFatherNodeId].m_nRightId, ptTrainData, nRightCnt, nFeatureCnt, pnSampleIdx + nLeftcnt, pnFeatureIdx, nCurDepth + 1);
}

void CCart::BestCut(CDataSet * ptTrainData, int nSampleCnt, int nFeatureCnt, int * pnSampleIdx, int * pnFeatureIdx, int & nThFeatureId, int & nThSampleIdx)
{
    double dMinInfo, dInfo;
    int nIdx;

    dMinInfo = 0;
    nThSampleIdx = -1;
    nThFeatureId = -1;

    for (int i = 0; i < nFeatureCnt; i++)
    {
        int nFeatureId = pnFeatureIdx[i];
        QSortSampleIdx(ptTrainData, nFeatureId, pnSampleIdx, 0, nSampleCnt - 1);

        switch (m_eLearnTarget)
        {
        case ELearnTarget_Classification:
            FindMinInfoC(ptTrainData, nFeatureId, pnSampleIdx, nSampleCnt, dInfo, nIdx);
            break;
        case ELearnTarget_Regression:
            FindMinInfoR(ptTrainData, nFeatureId, pnSampleIdx, nSampleCnt, dInfo, nIdx);
            break;
        }        

        if (i == 0 || dInfo < dMinInfo || (dInfo == dMinInfo && abs(nIdx - nSampleCnt / 2) < abs(nThSampleIdx - nSampleCnt / 2)))
        {
            dMinInfo = dInfo;
            nThSampleIdx = nIdx;
            nThFeatureId = nFeatureId;
        }
    }
    QSortSampleIdx(ptTrainData, nThFeatureId, pnSampleIdx, 0, nSampleCnt - 1);
}

void CCart::QSortSampleIdx(CDataSet * ptTrainData, int nFeatureId, int * pnIdx, int l, int r)
{
    int i, j, tmp;
    double t;
    TypeF *pdFeature;
    i = l; j = r;
    int nIdx = pnIdx[(i + j) >> 1];
    pdFeature = ptTrainData->GetFeature(nIdx);
    t = pdFeature[nFeatureId];

    do
    {
        pdFeature = ptTrainData->GetFeature(pnIdx[i]);
        while (pdFeature[nFeatureId] < t)
        {
            i++;
            pdFeature = ptTrainData->GetFeature(pnIdx[i]);
        }

        pdFeature = ptTrainData->GetFeature(pnIdx[j]);
        while (pdFeature[nFeatureId] > t)
        {
            j--;
            pdFeature = ptTrainData->GetFeature(pnIdx[j]);
        }

        if (i <= j)
        {
            tmp = pnIdx[i];
            pnIdx[i] = pnIdx[j];
            pnIdx[j] = tmp;
            i++; j--;
        }
    } while (i <= j);
    if (l < j) QSortSampleIdx(ptTrainData, nFeatureId, pnIdx, l, j);
    if (i < r) QSortSampleIdx(ptTrainData, nFeatureId, pnIdx, i, r);
}

double CCart::CalcInfo(double * pdHist, int nLen, int nInfoType)
{
    double dSum = 0;
    int i = 0;
    for (i = 0; i < nLen; i++) dSum += pdHist[i];
    double dInfo = 0.0f;
    if (dSum < 1e-15) return 0;
    dSum = 1.0f / dSum;
    for (i = 0; i < nLen; i++)
    {
        double p = pdHist[i] * dSum;
        switch (nInfoType)
        {
        case INFO_GINI:
            dInfo += p * p;
            break;
        case INFO_ENTROPY:
            dInfo -= p * log2(p + 1e-20);
            break;
        }
    }

    if (INFO_GINI == nInfoType)
    {
        dInfo = 1 - dInfo;
    }
    //dInfo = 1 - dInfo;
    return dInfo;
}

void CCart::FindMinInfoC(CDataSet * ptTrainSamples, int nFeatureId, int * pnSampleIdx, int nSampleCnt, double & dMinInfo, int & nIdx)
{
    double fInfo, dWeight;
    int i, nSampleId, nLabel, nLableIdx;
    int nMaxClassCnt = m_mLabelMap.size();
    vector<double> vdLHist(nMaxClassCnt, 0);
    vector<double> vdRHist(nMaxClassCnt, 0);

    //inital th position
    TypeF *pFeature1 = NULL;
    TypeF *pFeature2 = NULL;
    nIdx = -1;
    //for (nIdx = 0; nIdx < nSampleCnt - 1; nIdx++)
    //{
    //    nSampleId = pnSampleIdx[nIdx];
    //    nLabel = ptTrainSamples->GetLabel(nSampleId);
    //    nLableIdx = m_mLabelMap[nLabel];
    //    dWeight = ptTrainSamples->GetWeight(nSampleId);
    //    vdLHist[nLableIdx] += dWeight;

    //    pFeature1 = ptTrainSamples->GetFeature(pnSampleIdx[nIdx]);
    //    pFeature2 = ptTrainSamples->GetFeature(pnSampleIdx[nIdx + 1]);
    //    if (pFeature1[nFeatureId] != pFeature2[nFeatureId]) break;
    //}
    for (i = nIdx + 1; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        nLabel = ptTrainSamples->GetLabel(nSampleId);
        nLableIdx = m_mLabelMap[nLabel];
        dWeight = ptTrainSamples->GetWeight(nSampleId);
        vdRHist[nLableIdx] += dWeight;
    }

    //search one by one 
    if(m_nSplitRule == INFO_MINERRWEIGHT) dMinInfo = 1 - fabs(vdLHist[0] - vdLHist[1]);
    else dMinInfo = CalcInfo(&vdLHist[0], nMaxClassCnt, m_nSplitRule) * (nIdx + 1)+ CalcInfo(&vdRHist[0], nMaxClassCnt, m_nSplitRule) * (nSampleCnt - nIdx - 1);
    for (i = nIdx + 1; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        nLabel = ptTrainSamples->GetLabel(nSampleId);
        nLableIdx = m_mLabelMap[nLabel];
        dWeight = ptTrainSamples->GetWeight(nSampleId);
        vdLHist[nLableIdx] += dWeight;
        vdRHist[nLableIdx] -= dWeight;

        if (i < nSampleCnt - 1)
        {
            pFeature1 = ptTrainSamples->GetFeature(pnSampleIdx[i]);
            pFeature2 = ptTrainSamples->GetFeature(pnSampleIdx[i + 1]);
            //if the feature equals to the follow one, there shouldn't have a split
            if (pFeature1[nFeatureId] == pFeature2[nFeatureId]) continue;
        }

        if (m_nSplitRule == INFO_MINERRWEIGHT) fInfo = 1 - fabs(vdLHist[0] - vdLHist[1]);
        else fInfo = CalcInfo(&vdLHist[0], nMaxClassCnt, m_nSplitRule) * (i + 1) + CalcInfo(&vdRHist[0], nMaxClassCnt, m_nSplitRule) * (nSampleCnt - i - 1);
        if ((fInfo < dMinInfo) || (fInfo == dMinInfo && abs(i - nSampleCnt/2) < abs(nIdx - nSampleCnt / 2)))
        {
            dMinInfo = fInfo;
            nIdx = i;
        }
    }
}

bool CCart::IsNodePureC(CDataSet * ptTrainData, int nSampleCnt, int * pnSampleIdx)
{
    int i, nSampleId;
    if (nSampleCnt <= 1) return true;

    nSampleId = pnSampleIdx[0];
    int nPreLabel, nCurLabel;
    nPreLabel = ptTrainData->GetLabel(nSampleId);
    for (i = 1; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        nCurLabel = ptTrainData->GetLabel(nSampleId);
        if (nCurLabel != nPreLabel) return false;
        nCurLabel = nPreLabel;
    }

    return true;
}

void CCart::GenLeafNodeC(TTreeNode * pNode, CDataSet * ptTrainData, int nSampleCnt, int * pnSampleIdx)
{
    int i, nIdx;
    int nSampleId;
    int nLabel;
    double dWeight;
    pNode->m_nLeftId = pNode->m_nRightId = -1;

    vector<double> vdHist(m_mLabelMap.size(), 0);
    for (i = 0; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        nLabel = ptTrainData->GetLabel(nSampleId);
        nIdx = m_mLabelMap[nLabel];
        dWeight = ptTrainData->GetWeight(nSampleId);
        vdHist[nIdx] += dWeight;
    }
    int nMaxWeightId = 0;
    double fMaxWeight = 0;
    double fSumWeight = 0;
    for (i = 0; i < (int)vdHist.size(); i++)
    {
        fSumWeight += vdHist[i];
        if (vdHist[i] > fMaxWeight)
        {
            nMaxWeightId = i;
            fMaxWeight = vdHist[i];
        }
    }
    pNode->m_nLabel = m_vLabelMapInv[nMaxWeightId];//to do. find the original label nMaxWeightId;//
    pNode->m_dCorrectW = fMaxWeight;
    pNode->m_dErrorW = fSumWeight - fMaxWeight;
    pNode->m_fConfidence = fMaxWeight / fSumWeight;

    //map the prob 
    if (m_pfProbMap) pNode->m_fConfidence = m_pfProbMap(pNode->m_nLabel, pNode->m_fConfidence);
}

void CCart::FindMinInfoR(CDataSet * ptTrainSamples, int nFeatureId, int * pnSampleIdx, int nSampleCnt, double & dMinInfo, int & nIdx)
{
    int nResponseLen = ptTrainSamples->GetResponseDim();
    vector<double> pLSum0(nResponseLen, 0);
    vector<double> pLSum1(nResponseLen, 0);
    vector<double> pRSum0(nResponseLen, 0);
    vector<double> pRSum1(nResponseLen, 0);
    double temp, temp1, temp2;
    int i, j, k;
    int nSampleId = pnSampleIdx[0];
    TypeR *pResponse;

    //inital th position
    TypeF *pFeature1 = NULL;
    TypeF *pFeature2 = NULL;
 /*   for (nIdx = 0; nIdx < nSampleCnt; nIdx++)
    {
        nSampleId = pnSampleIdx[nIdx];
        pResponse = ptTrainSamples->GetResponse(nSampleId);
        for (j = 0; j < nResponseLen; j++)
        {
            temp = pResponse[j];
            pLSum0[j] += temp;
            pLSum1[j] += temp * temp;
        }

        pFeature1 = ptTrainSamples->GetFeature(pnSampleIdx[nIdx]);
        pFeature2 = ptTrainSamples->GetFeature(pnSampleIdx[nIdx + 1]);
        if (nIdx < nSampleCnt - 1 && pFeature1[nFeatureId] != pFeature2[nFeatureId]) break;
    }*/
    nIdx = -1;
    for (i = nIdx + 1; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        pResponse = ptTrainSamples->GetResponse(nSampleId);
        for (j = 0; j < nResponseLen; j++)
        {
            temp = pResponse[j];
            pRSum0[j] += temp;
            pRSum1[j] += temp * temp;
        }
    }

    dMinInfo = 0;
    temp1 = nIdx + 1;
    temp2 = nSampleCnt - nIdx - 1;
    for (k = 0; k < nResponseLen; k++)
    {
        double fLVarN = temp1 * pLSum1[k] - pLSum0[k] * pLSum0[k];
        double fRVarN = temp2 * pRSum1[k] - pRSum0[k] * pRSum0[k];
        dMinInfo += fLVarN + fRVarN;
    }

    //search
    for (i = nIdx + 1; i < nSampleCnt - 1; i++)
    {
        nSampleId = pnSampleIdx[i];
        pResponse = ptTrainSamples->GetResponse(nSampleId);
        for (j = 0; j < nResponseLen; j++)
        {
            temp = pResponse[j];
            pLSum0[j] += temp;
            pRSum0[j] -= temp;
            temp = temp * temp;
            pLSum1[j] += temp;
            pRSum1[j] -= temp;
        }

        pFeature1 = ptTrainSamples->GetFeature(pnSampleIdx[i]);
        pFeature2 = ptTrainSamples->GetFeature(pnSampleIdx[i + 1]);
        //if the feature equals to the follow one, there shouldn't have a split
        if (pFeature1[nFeatureId] == pFeature2[nFeatureId]) continue;

        double fVar = 0;
        temp1 = i + 1;
        temp2 = nSampleCnt - i - 1;
        for (k = 0; k < nResponseLen; k++)
        {
            double fLVarN = temp1 * pLSum1[k] - pLSum0[k] * pLSum0[k];
            double fRVarN = temp2 * pRSum1[k] - pRSum0[k] * pRSum0[k];
            fVar += fLVarN + fRVarN;
        }
        if (fVar < dMinInfo)
        {
            dMinInfo = fVar;
            nIdx = i;
        }
    }
}

bool CCart::IsNodePureR(CDataSet * ptTrainData, int nSampleCnt, int * pnSampleIdx)
{
    if (nSampleCnt <= 1) return true;
    return false;
}

void CCart::GenLeafNodeR(TTreeNode * pNode, CDataSet * ptTrainData, int nSampleCnt, int * pnSampleIdx)
{
    int i, j;
    int nSampleId;
    int nResponseLen = ptTrainData->GetResponseDim();
    pNode->m_nLeftId = pNode->m_nRightId = -1;

    pNode->m_vResponse = vector<double>(nResponseLen, 0);
    for (i = 0; i < nSampleCnt; i++)
    {
        nSampleId = pnSampleIdx[i];
        TypeR *pResponse = ptTrainData->GetResponse(nSampleId);
        for (j = 0; j < nResponseLen; j++)
            pNode->m_vResponse[j] += pResponse[j];
    }

    for (j = 0; j < nResponseLen; j++)
        pNode->m_vResponse[j] /= nSampleCnt;
}

TTreeNode * CCart::FindLeafNode(TypeF *pdFeature, int & nLeafNodeId)
{
    TTreeNode *pNode = &m_vtNode[0];
    int nCurRightMinId = 1 << (m_nMaxDepth - 1);
    while (pNode->m_nLeftId > 0)
    {
        if (pdFeature[pNode->m_nIdx] <= pNode->m_FeatureTh)
        {
            pNode = &m_vtNode[pNode->m_nLeftId];
        }
        else
        {
            pNode = &m_vtNode[pNode->m_nRightId];
            nLeafNodeId += nCurRightMinId;
        }
        nCurRightMinId >>= 1;
    }
    return pNode;
}
