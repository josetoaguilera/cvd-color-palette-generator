from setuptools import setup, find_packages

setup(
    name='cvd_color_palette_generator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'colormath>=3.0.0',
        'daltonlens>= 0.1.5',
        'matplotlib>=3.9.0',
        'scikit-image>=0.24.0',
        'plotly>=5.23.0',
        'pandas>=2.2.2',
        'ipywidgets==7.7.1',
        'opencv-python>=4.10.0.84',
        'scikit-learn>=1.5.1'        
    ],
    author='Jose Tomas Aguilera Yevenes',
    author_email='joseaguilera@ug.uchile.cl',
    description='librería para generar paletas de colores para personas con discapacidad visual',
    url='https://github.com/josetoaguilera/cvd-color-palette-generator',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
