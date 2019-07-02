from setuptools import setup


requires = [
            "scipy>=1.3.0",
            "numpy>=1.16.1",
            "librosa>=0.6.3",
            "wavio>=0.0.4",
            ]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='prowav',
<<<<<<< HEAD
    version='0.4',
=======
    version='0.3',
>>>>>>> ver-0.3
    description='The package for preprocessing wave data',
    url='https://github.com/wildgeece96/prowav',
    author='Soh',
    author_email='wildgeece96@gmail.com',
    license='MIT',
    keywords='wave mfcc fft',
    packages=[
        "prowav",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
