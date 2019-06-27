from setuptools import setup


requires = [
            "scipy>=1.2.1",
            "numpy>=1.16.1",
            "librosa>=0.6.3",
            "wavio>=0.0.4"
            ]


setup(
    name='prowav',
    version='0.3',
    description='The package for preprocessing wave data',
    url='https://github.com/wildgeece96/prowav',
    author='Soh',
    author_email='wildgeece96@gmail.com',
    license='MIT',
    keywords='wave mfcc fft',
    packages=[
        "prowav",
    ],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
