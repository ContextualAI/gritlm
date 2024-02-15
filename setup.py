from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='gritlm',
    version='0.9.9',
    description='gritlm',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='n.muennighoff@gmail.com',
    url='https://github.com/ContextualAI/gritlm',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2.0',
        'accelerate>=0.26.1',
        'transformers>=4.37.2',
        'datasets>=2.16.1',
        'wandb',
        'mteb[beir]'
    ],
)
