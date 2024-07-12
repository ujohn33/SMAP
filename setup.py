from setuptools import setup, find_packages

setup(
    name='SMAP',  # Replace with the name of your library
    version='0.1.0',  # Replace with your library's version
    author='Evgenii Genov',  # Replace with your name
    author_email='eugengenov@gmail.com',  # Replace with your email address
    description='This libary is a recreation of an R package for smart meter alanlysis of consumption and temperature time-series. ',  # Replace with a brief description of your library
    long_description=open('README.md').read(),  # Long description read from the the readme file
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/SMAP',  # Replace with the link to your github or project website
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'polars>=0.20.31'  # Replace with the version of Polars you are using or compatible with

    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Change as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',  # Replace with the Python version requirements
    keywords='meter analytics polars',  # Replace with keywords relevant to your library
)
