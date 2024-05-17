from setuptools import setup, find_packages

setup(
    name="rotation_steerers",
    packages=find_packages(include=[
        "rotation_steerers*"
    ]),
    install_requires=[
        "DeDoDe",
    ],
    python_requires='>=3.9.0',
    version="0.0.2",
    author="Georg Bökman",
)
