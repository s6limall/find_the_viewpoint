#include "../include/ApplyLightGlueWrapper.hpp"
#include <iostream>

ApplyLightGlueWrapper::ApplyLightGlueWrapper() : pClass(nullptr), pInstance(nullptr) {}

bool ApplyLightGlueWrapper::initialize() {
    Py_Initialize();
    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString("./LightGlue"));  // Adjust to your actual path

    PyObject* pName = PyUnicode_DecodeFSDefault("apply_lightglue"); // Replace with the actual module name
    if (!pName) {
        PyErr_Print();
        std::cerr << "Failed to decode module name" << std::endl;
        return false;
    }

    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule) {
        PyErr_Print();
        std::cerr << "Failed to import module" << std::endl;
        return false;
    }

    pClass = PyObject_GetAttrString(pModule, "LightGlueWrapper");
    Py_DECREF(pModule);

    if (!pClass || !PyCallable_Check(pClass)) {
        PyErr_Print();
        std::cerr << "Failed to get class or class is not callable" << std::endl;
        Py_XDECREF(pClass);
        Py_Finalize();
        return false;
    }

    pInstance = PyObject_CallObject(pClass, nullptr);
    if (!pInstance) {
        PyErr_Print();
        std::cerr << "Failed to create an instance of the class" << std::endl;
        Py_XDECREF(pClass);
        Py_Finalize();
        return false;
    }

    return true;
}

double ApplyLightGlueWrapper::apply(const std::string& image_path1, const std::string& image_path2) {
    if (!pInstance) {
        std::cerr << "pInstance Error: Instance not initialized" << std::endl;
        return -1.0; // Initialization check
    }

    PyObject* pFunc = PyObject_GetAttrString(pInstance, "apply");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        std::cerr << "pFunc Error: Method 'apply' not found or not callable" << std::endl;
        Py_XDECREF(pFunc);
        return -1.0;
    }

    PyObject* pArgs = PyTuple_Pack(2, PyUnicode_FromString(image_path1.c_str()), PyUnicode_FromString(image_path2.c_str()));
    if (!pArgs) {
        PyErr_Print();
        std::cerr << "Failed to create argument tuple" << std::endl;
        Py_XDECREF(pFunc);
        return -1.0;
    }

    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);

    if (pValue != nullptr) {
        double result = PyFloat_AsDouble(pValue);
        Py_DECREF(pValue);
        std::cerr << "Function executed and returned correctly" << std::endl;
        return result;
    } else {
        PyErr_Print();
        std::cerr << "pValue Error: Failed to get a return value" << std::endl;
        return -1.0;
    }
}

void ApplyLightGlueWrapper::finalize() {
    Py_XDECREF(pInstance);
    Py_XDECREF(pClass);
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}

ApplyLightGlueWrapper::~ApplyLightGlueWrapper() {
    finalize();
}
