try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='estimator-optimize',
      version='1.4.0',
      description='estimator optimization toolbox.',
      license='BSD',
      author='zuo.wang@sky-data.cn',
      packages=['estimator_optimize', 'estimator_optimize.learning', 
                'estimator_optimize.optimizer', 'estimator_optimize.space',
                'estimator_optimize.learning.gaussian_process'],
      install_requires=["numpy", "scipy>=0.14.0", "scikit-learn>=0.19.1",
                        "matplotlib"]
      )
