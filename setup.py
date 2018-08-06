from setuptools import setup

setup(
    name='wavetable',
    version='',
    packages=['wavetable', 'wavetable.util'],
    url='',
    license='',
    author='nyanpasu64',
    author_email='',
    description='',
    install_requires=['numpy', 'scipy', 'ruamel.yaml>=0.15.0', 'waveform_analysis', 'click',
                      'dataclasses;python_version<"3.7"'],
    dependency_links=[
        'git+https://github.com/endolith/waveform_analysis.git@master'
    ]
)
