from setuptools import setup, find_packages

setup(
    name='wavetable',
    version='0.1.0',
    packages=find_packages('.'),
    url='',
    license='',
    author='nyanpasu64',
    author_email='',
    description='',
    install_requires=['numpy', 'scipy', 'ruamel.yaml>=0.15.0', 'waveform_analysis @ git+https://github.com/endolith/waveform_analysis.git@master', 'click',
                      'dataclasses;python_version<"3.7"'],
    entry_points = {
        'console_scripts': [
            'to-brr=wavetable.to_brr:main',
            'wave_reader=wavetable.wave_reader:main',
        ],
    }
)
