from setuptools import setup, find_packages

setup(
    name="unimatch",
    version="0.1.0",
    description="Unifying Flow, Stereo and Depth Estimation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Xu, Haofei and Zhang, Jing and Cai, Jianfei and Rezatofighi, Hamid and Yu, Fisher and Tao, Dacheng and Geiger, Andreas",
    url="https://haofeixu.github.io/unimatch/",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
        "einops",
        "opencv-python",
        "matplotlib",
        "tensorboard==2.9.1",
        "imageio==2.9.0",
        "imageio-ffmpeg",
        "pillow",
        "scikit-image",
        "scipy",
    ],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)