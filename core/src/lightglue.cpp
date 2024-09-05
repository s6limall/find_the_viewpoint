#ifndef LIGHTGLUEWRAPPER_HPP
#define LIGHTGLUEWRAPPER_HPP

#include <Python.h>

class LightGlueWrapper {
public:
    // Constructor: Initializes member variables but does not initialize Python environment or load module
    LightGlueWrapper();

    // Destructor: Cleans up Python objects and finalizes the Python environment
    ~LightGlueWrapper();

    // Method to initialize the Python environment and load the module
    bool initialize();

    // Method to compute the match ratio using the LightGlue Python function
    double compute_match_ratio_LIGHTGLUE();

private:
    PyObject* pName;    // Python string for module name
    PyObject* pModule;  // Python object for the loaded module
    PyObject* pFunc;    // Python object for the function to call
};

#endif // LIGHTGLUEWRAPPER_HPP
