# setup.py
from setuptools import setup, find_packages

setup(
    name="caika-rag-typhoon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain_community",
        "sentence-transformers",
        "faiss-cpu",
        "pypdf",
        "tiktoken",
        "llama-cpp-python",
        "streamlit",
    ],
)