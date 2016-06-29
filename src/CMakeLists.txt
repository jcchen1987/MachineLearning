set(SRC_LIB ml_AdaBoost.cpp  ml_Cart.cpp  ml_Crossvalidate.cpp  ml_Forest.cpp  ml_Learner.cpp)
add_library(libml SHARED ${SRC_LIB})
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set_target_properties(libml PROPERTIES OUTPUT_NAME "ml")
