from setuptools import find_packages, setup

setup(name='MDSimsEval',
      version='0.0.1-dev',
      description='Evaluation of a collection of MD simulations.',
      install_requires=[
            'tqdm>=4.42.0',
            'numpy>=1.18.1',
            'pandas>=1.0.3',
            'matplotlib>=3.1.2',
            'imgkit>=1.0.2',
            'scipy>=1.4.1',
            'MDAnalysis>=0.20.1',
            'mdtraj>=1.9.3',
            'scikit-learn>=0.22.1',
            'seaborn>=0.10.0'
      ],
      packages=find_packages(),
      zip_safe=False)
