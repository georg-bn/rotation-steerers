from setuptools import setup, find_packages

setup(
    name="rotation_steerers",
    packages=find_packages(include=[
        "rotation_steerers*"
    ]),
    install_requires=[
        "DeDoDe @ git+https://github.com/Parskatt/DeDoDe.git@dedode_pretrained_models",
    ],
    python_requires='>=3.9.0',
    version="0.0.1",
    author="Georg Bökman",
)
