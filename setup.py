import setuptools

setuptools.setup(
    name="tts_webui_extension.audiocraft_mac",
    packages=setuptools.find_namespace_packages(),
    version="0.0.8",
    author="rsxdalv",
    description="Audiocraft fork for Apple Silicon",
    url="https://github.com/rsxdalv/tts_webui_extension.audiocraft_mac",
    project_urls={},
    scripts=[],
    install_requires=[
        "audiocraft_apple_silicon @ git+https://github.com/rsxdalv/audiocraft@audiocraft_apple_silicon_ext",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
