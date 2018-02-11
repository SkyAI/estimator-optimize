Estiamtor-Optimize
==================

Estiamtor-Optimize is a auto hyper paramater tune library.

The library is built on top of NumPy, SciPy, Scikit-Learn and TensorFlow.

Install
-------
::
    python install -r requirements.txt
    python setup.py install


Getting started
---------------
::
    python benchmarks/benchmark_classify.py
    python benchmarks/tf_classify.py

Results
-------
baseline:0.953704

|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| forest_minimize | -0.967 +/- 0.006 | -0.976 | 18.200 | 10.332 | 2.000
------------------|------------|-----------|---------------------|--------------------|-----------------------

| gbrt_minimize | -0.968 +/- 0.005 | -0.976 | 15.900 | 15.202 | 0.000
------------------|------------|-----------|---------------------|--------------------|-----------------------

| gp_minimize | -0.971 +/- 0.005 | -0.981 | 34.500 | 10.404 | 15.000
------------------|------------|-----------|---------------------|--------------------|-----------------------

Credits
-------
this repo uses code from scikit-optimize[http://github.com/scikit-optimize/scikit-optimize]