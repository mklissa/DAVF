from setuptools import setup
from setuptools import find_packages

setup(name='davf',
      version='1.0',
      description='Diffusion-Based Approximat Value Functions',
      author='Martin Klissarov',
      author_email='martin.klissarov@mail.mcgill.ca',
      download_url='https://github.com/mklissa/DAVF',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'networkx',
                        'scipy',
                        'gym==0.9.3',
                        'mujoco-py==0.5.7',
                        'opencv-python',
                        'sklearn'
                        ],
      packages=find_packages())