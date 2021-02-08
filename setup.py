from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("apollo/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name ='apollo',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stan0625@uw.com',
    url = 'git@github.com:stanbiryukov/apollo.git',
    install_requires = requirements,
    dependency_links=['http://github.com/stanbiryukov/PyTorch-LBFGS/tarball/master#egg=torchlbfgs'],
    package_data = {'apollo':['resources/*']},
    packages = find_packages(exclude=['apollo/tests']),
    license = 'MIT',
    description='Apollo: Performant out-of-the-box GP regression and classification based on GPyTorch, with a familiar sklearn api',
    long_description= "Apollo is a batteries included GP regression and classification model.",
    keywords = ['statistics','classification','regression', 'gp', 'gaussian-process', 'GPyTorch', 'pytorch', 'machine-learning', 'scikit-learn'],
    classifiers = [
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    **extra_setuptools_args
)