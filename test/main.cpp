#include <stdio.h>
#include <stdlib.h>
#include "ml_DataSet.h"
#include "ml_Learner.h"
#include "ml_Crossvalidate.h"

#define TEST_R 0
void LearnerTest(ELearner eLearner);

int main()
{  
    //ELearner_Forest
    //ELearner_Cart
    //ELearner_AdaBoost
    LearnerTest(ELearner_AdaBoost);
    
    return 0;
}


struct TSample
{
    vector<TypeF> vFeature;
    vector<TypeR> vResponse;
    double dWeight;
    int nLabel;
};

class CMyDataSet : public CDataSet
{
public:
    CMyDataSet()
    {

    }

    void Load(int nCnt, bool bRegression)
    {
        m_vtSample.resize(nCnt);
        for (int i = 0; i < (int)m_vtSample.size(); i++)
        {
            double x = rand() * 2.0 / RAND_MAX - 1;
            double y = rand() * 2.0 / RAND_MAX - 1;
            m_vtSample[i].vFeature.resize(2);
            m_vtSample[i].vFeature[0] = x;
            m_vtSample[i].vFeature[1] = y;
            m_vtSample[i].dWeight = 1;

            if (bRegression)
            {
                m_vtSample[i].vResponse.push_back(x + y);
            }
            else
            {
                if (x*x + y*y < 0.25)
                {
                    m_vtSample[i].nLabel = 1;
                }
                else
                {
                    m_vtSample[i].nLabel = -1;
                }
            }     
        }
    }

    virtual int GetSampleCnt()
    {
        return m_vtSample.size();
    }
    virtual int GetFeatureDim()
    {
        if (m_vtSample.empty()) return 0;
        return m_vtSample[0].vFeature.size();
    }
    virtual int GetResponseDim()
    {
        if (m_vtSample.empty()) return 0;
        return m_vtSample[0].vResponse.size();
    }
    virtual TypeF * GetFeature(int nSampleId)
    {
        if (nSampleId < 0 || nSampleId >= (int)m_vtSample.size() || m_vtSample[nSampleId].vFeature.empty()) return NULL;
        return &m_vtSample[nSampleId].vFeature[0];
    }
    virtual TypeR * GetResponse(int nSampleId)
    {
        if (nSampleId < 0 || nSampleId >= (int)m_vtSample.size() || m_vtSample[nSampleId].vResponse.empty()) return NULL;
        return &m_vtSample[nSampleId].vResponse[0];
    }
    virtual int GetLabel(int nSampleId)
    {
        if (nSampleId < 0 || nSampleId >= (int)m_vtSample.size()) return 0;
        return m_vtSample[nSampleId].nLabel;
    }
    virtual double GetWeight(int nSampleId)
    {
        if (nSampleId < 0 || nSampleId >= (int)m_vtSample.size()) return -1;
        return m_vtSample[nSampleId].dWeight;
    }
    virtual void SetWeight(int nSampleId, double dWeight)
    {
        if (nSampleId < 0 || nSampleId >= (int)m_vtSample.size()) return;
        m_vtSample[nSampleId].dWeight = dWeight;
    }
private:
    vector<TSample> m_vtSample;
};


void LearnerTest(ELearner eLearner)
{
    bool bRegression = false;
    int nSampleCnt = 2000;
    CMyDataSet *pDataSet = new CMyDataSet;
    pDataSet->Load(nSampleCnt, bRegression);
    CLearner *pLearner = CreateLearner(eLearner);
    CCrossValidation crossValidation(K_FOLDER, 2);// (LEAVE_ONE_OUT);
    ELearnTarget eTar = bRegression ? ELearnTarget_Regression : ELearnTarget_Classification;

    TCartParam tCartParam;
    tCartParam.pfProbMapFun = NULL;
    tCartParam.nMaxTreeDepth = 5;
    tCartParam.eSplitRule = ESplitRule_GINI;
    tCartParam.eLearnTarget = eTar;

    TForestParam tForestParam;
    tForestParam.nMaxTreeCnt = 10;
    tForestParam.nMaxTreeDepth = 4;
    tForestParam.eLearnTarget = eTar;
    tForestParam.dSelectSampleRate = 1;
    tForestParam.dSelectFeatureRate = 1;
    tForestParam.bRestartTrain = true;

    TAdaBoostParam tAdaBoostParam;
    tAdaBoostParam.nMaxTreeCnt = 4;
    tAdaBoostParam.nMaxTreeDepth = 2;
    tAdaBoostParam.eBoostType = EBoostType_Gentle; //EBoostType_Discrete /  EBoostType_Real / EBoostType_Gentle
    tAdaBoostParam.bRestartTrain = true;
    tAdaBoostParam.bSoftCascade = false;
    tAdaBoostParam.dRecall = 0.99;

    void *pvParam = NULL;
    switch (eLearner)
    {
    case ELearner_Cart:
        pvParam = (void *)&tCartParam;
        break;
    case ELearner_Forest:
        pvParam = (void *)&tForestParam;
        break;
    case ELearner_AdaBoost:
        pvParam = (void *)&tAdaBoostParam;
        break;
    }
    pLearner->SetParam(pvParam);

    double dErr = 1;
    crossValidation.Validate(pLearner, pDataSet, dErr);
    printf("err = %lf\n", dErr);

    DestroyLearner(pLearner);
}
