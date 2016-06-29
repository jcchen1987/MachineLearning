/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_CART_H__
#define __ML_CART_H__
#include "ml_DataSet.h"
#include "ml_Learner.h"
#include <stdio.h>
#include <vector>
#include <map>
using std::vector;
using std::map;

struct TTreeNode
{
    //middle node
    int    m_nIdx;
    TypeF  m_FeatureTh;
    int    m_nLeftId;
    int    m_nRightId;

    //leaf node
    vector<TypeR> m_vResponse;

    int    m_nLabel;
    double m_dCorrectW;
    double m_dErrorW;
    double m_fConfidence;

    void Save(FILE *&pf);
    void Load(FILE *&pf);
};

class CCart : public CLearner
{
public:
    CCart();
    ~CCart();
    virtual int SetParam(void *pvParam);
    virtual int Train(CDataSet *pDataSet, int *pnSampleIdx = NULL, int nSampleCnt = 0, int *pnFeatureIdx = NULL, int nFeatureCnt = 0);
    virtual int Predict(TypeF *pFeature, int &nLabel, double &dProb);
    virtual int Predict(TypeF *pFeature, TypeR *pResponse);
    virtual int Save(FILE *&pf);
    virtual int Load(FILE *&pf);
    virtual int GetLearnTargetType(ELearnTarget &eTarget);

private:
    //common functions
    void InitIndex(vector<int> &vnSampleIdx, vector<int> &vnFeatureIdx, int nAllSmpCnt, int nAllFeaCnt, int * pnSampleIdx, int nSampleCnt, int * pnFeatureIdx, int nFeatureCnt);
    void SplitNode(int nFatherNodeId, CDataSet *ptTrainData, int nSampleCnt, int nFeatureCnt, int *pnSampleIdx, int *pnFeatureIdx, int nCurDepth);
    void BestCut(CDataSet *ptTrainData, int nSampleCnt, int nFeatureCnt, int *pnSampleIdx, int *pnFeatureIdx, int &nThFeatureId, int &nThSampleIdx);
    void QSortSampleIdx(CDataSet *ptTrainData, int nFeatureId, int *pnIdx, int l, int r);
    TTreeNode* FindLeafNode(TypeF *pdFeature, int &nLeafNodeId);

    //only for classification
    void MapIdx(CDataSet *ptTrainData, int *pnSampleIdx, int nSampleCnt);
    double CalcInfo(double * pdHist, int nLen, int nInfoType);
    void FindMinInfoC(CDataSet *ptTrainSamples, int nFeatureId, int *pnSampleIdx, int nSampleCnt, double &dMinEntropy, int &nIdx);
    bool IsNodePureC(CDataSet *ptTrainData, int nSampleCnt, int *pnSampleIdx);
    void GenLeafNodeC(TTreeNode *pNode, CDataSet *ptTrainData, int nSampleCnt, int *pnSampleIdx);

    //only for regression
    void FindMinInfoR(CDataSet *ptTrainSamples, int nFeatureId, int *pnSampleIdx, int nSampleCnt, double &dMinEntropy, int &nIdx);
    bool IsNodePureR(CDataSet *ptTrainData, int nSampleCnt, int *pnSampleIdx);
    void GenLeafNodeR(TTreeNode *pNode, CDataSet *ptTrainData, int nSampleCnt, int *pnSampleIdx);


private:
    ELearnTarget m_eLearnTarget;
    int m_nMaxDepth;
    int m_nSplitRule;
    vector<TTreeNode> m_vtNode;

    //the info below is not need to save
    map<int, int>           m_mLabelMap;    //Label -> [0,c-1]
    vector<int>             m_vLabelMapInv; //[0,c-1] -> Label 
    PFProbMap m_pfProbMap;
    int m_nLeafNodeId;
};

#endif

