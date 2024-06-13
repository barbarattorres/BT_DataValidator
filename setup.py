from setuptools import setup, find_packages

setup(
    name='data_validator',  
    version='0.1.0',  
    author='Barbara Torres', 
    author_email='barbaratt@icloud.com',  
    description='A sophisticated tool for comprehensive data validation within pandas DataFrames.',
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',  
    url='https://github.com/barbarattorres/BT_DataValidator', 
    packages=find_packages(), 
    install_requires=[
        'pandas>=1.1',
        'numpy>=1.19',
        'python-dateutil>=2.8.1',
        'IPython>=7.0'     
    ], 
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',  
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimum version requirement of Python
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/barbarattorres/BT_DataValidator/issues',
        'Source': 'https://github.com/barbarattorres/BT_DataValidator',
    },
)
