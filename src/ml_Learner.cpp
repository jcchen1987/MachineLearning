#include "ml_Learner.h"
#include "ml_Cart.h"
#include "ml_Forest.h"
#include "ml_AdaBoost.h"

CLearner * CreateLearner(ELearner eLearner)
{
    CLearner *pLearner = NULL;
    switch (eLearner)
    {
    case ELearner_Cart:
        pLearner = new CCart;
        break;
    case ELearner_Forest:
        pLearner = new CForest;
        break;
    case ELearner_AdaBoost:
        pLearner = new CAdaBoost;
        break;
    }     
    return pLearner;
}

void DestroyLearner(CLearner * pLearner)
{
    if (pLearner)
    {
        delete pLearner;
    }
}
