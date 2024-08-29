#ifndef APPLY_LIGHTGLUE_WRAPPER_H
#define APPLY_LIGHTGLUE_WRAPPER_H

#include <Python.h>
#include <string>

class ApplyLightGlueWrapper {
public:
    ApplyLightGlueWrapper();
    bool initialize();
    double apply(const std::string& image_path1, const std::string& image_path2);
    void finalize();
    ~ApplyLightGlueWrapper();

private:
    PyObject* pClass;
    PyObject* pInstance;
};

#endif // APPLY_LIGHTGLUE_WRAPPER_H
