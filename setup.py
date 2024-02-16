from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='gritlm',
    version='0.9.1',
    description='GritLM',
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords="text generation, text embeddings, instruction tuning",
    license="Apache",
    author='Niklas Muennighoff',
    author_email='n.muennighoff@gmail.com',
    project_urls={
        "Huggingface Organization": "https://huggingface.co/gritlm",
        "Source Code": "https://github.com/ContextualAI/gritlm",
    },
    packages=find_packages(),
    python_requires=">=3.7.0",
    install_requires=[
        'accelerate>=0.26.1',
        'transformers>=4.37.2',
        'datasets>=2.16.1',
        'wandb',
        'mteb[beir]'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],    
)
