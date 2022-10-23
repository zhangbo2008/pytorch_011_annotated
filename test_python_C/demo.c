// 包含Python头文件
#include <Python.h>


// 兼容Python3
#if PY_MAJOR_VERSION >= 3
#define PYTHON3
#endif


// hello函数实现
static PyObject* hello(PyObject *self, PyObject *args)
{
    printf("Hello World\n");
    return Py_None;
}


// add函数实现
static PyObject* add(PyObject *self, PyObject *args)
{
    int a, b;
    if(!PyArg_ParseTuple(args, "ii", &a, &b))
    {
        return NULL;
    }
    return Py_BuildValue("i", a + b);
}


// 模块方法表
static PyMethodDef TwoMethods[] = {
    { "hello", hello, METH_NOARGS, "Print Hello" },
    { "add", add, METH_VARARGS, "Add two integers"},
    { NULL, NULL, 0, NULL }
};


#ifdef PYTHON3
// Python3模块定义结构体
static struct PyModuleDef testModule = {
    PyModuleDef_HEAD_INIT,
    "testModule",
    "Test Module",
    -1,
    TwoMethods
};


// Python3模块初始化函数
PyMODINIT_FUNC PyInit_demo(void)
{
    return PyModule_Create(&testModule);
}


#else
// Python2模块初始化函数
PyMODINIT_FUNC initdemo(void)
{
    Py_InitModule("demo", TwoMethods);
}
#endif