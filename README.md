# extension_audiocraft_mac is an extension for TTS-Generation-WebUI that allows generating music using Audiocraft.

Audiocraft (Fork, works with Apple Silicon)
Original fork: https://github.com/trizko/audiocraft Original repo issue about Apple M1, Silicon support: facebookresearch#31

This repo is a fork that works with Apple Silicon devices (e.g. M1 Macbook Pro). The original repo can be found here: https://github.com/facebookresearch/audiocraft

Installation
To install, you need to slightly modify the pip install command. Clone this repository and cd into the root directory of the repo. Then run the following commands:

# creating a virtual environment is optional (but recommended).
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# this step is required.
pip install --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu -e .
Running MusicGen
To run the application, you must set the PYTORCH_ENABLE_MPS_FALLBACK environment variable when running like so:

PYTORCH_ENABLE_MPS_FALLBACK=1 python app

# Changelog

* Make model lazy loaded
* Add model unload button
