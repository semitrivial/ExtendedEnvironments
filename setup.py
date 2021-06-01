import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name='extended_rl',
    version='0.1.0',
    author='double_blind',
    author_email='double_blind@donotemail.com',
    description='Extended Environments for Measuring Self-Reflection in RL',
    long_description=long_description,
    url='todo',
    packages=setuptools.find_packages(),
    python_requires='>=3.6.*',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
