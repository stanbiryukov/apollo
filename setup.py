from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("apollo/version.py") as f:
    exec(f.read())

extra_setuptools_args = dict(
    tests_require=['pytest']
)

required = []
dependency_links = []

# git installs too
EGG_MARK = '#egg='
for line in requirements:
    if line.startswith('-e git:') or line.startswith('-e git+') or \
            line.startswith('git:') or line.startswith('git+'):
        if EGG_MARK in line:
            package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
            required.append(package_name)
            dependency_links.append(line)
        else:
            print('Dependency to a git repository should have the format:')
            print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
    else:
        required.append(line)

setup(
    name ='apollo',
    version = __version__,
    author ='Stan Biryukov',
    author_email ='stan0625@uw.com',
    url = 'git@github.com:stanbiryukov/apollo.git',
    install_requires=required,
    dependency_links=dependency_links,
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