/*
* Author: jcchen
* Please contact jcchen1987@163.com
* if you have any questions or advices about this code
*/

#ifndef __ML_TRAINDATA_H__
#define __ML_TRAINDATA_H__
#include <vector>
using std::vector;

#if defined _WIN32
#define ML_EXPORT __declspec(dllexport)
#else
#define ML_EXPORT
#endif

typedef double TypeF;
typedef double TypeR;
class ML_EXPORT CDataSet
{
public:
    virtual int GetSampleCnt() = 0;
    virtual int GetFeatureDim() = 0;
    virtual int GetResponseDim() = 0;
    virtual TypeF * GetFeature(int nSampleId) = 0;
    virtual TypeR * GetResponse(int nSampleId) = 0;
    virtual int GetLabel(int nSampleId) = 0;
    virtual double GetWeight(int nSampleId) = 0;
    virtual void SetWeight(int nSampleId, double dWeight) = 0;
};

#endif // !__ML_TRAINDATA_H__
