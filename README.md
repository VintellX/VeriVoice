```bash
git clone --recursive https://github.com/VintellX/VeriVoice.git
cd VeriVoice
```

## Overview
VeriVoice is a DeepFake Audio Detection model, this is a PyTorch-based framework for detecting cloned or syntthesized speech. It uses a Graph-Enhanced Audio Transformer to capture both local spectral artifacts and global temporal patterns that distinguish real voices from their deepfakes or clones.

## Quick Start
1. **Install dependencies**:
    ```bash
    conda create -n VeriVoice python=3.10
    conda activate VeriVoice
    pip install -r requirements.txt
    ```
2. **Prepare data** (check out `usage.md` for details).
3. **Train the mdoel**:
    ```bash
    python -m ipykernel install --user --name=VeriVoice --display-name "VeriVoice-Env"
    jupyter notebook
    ```
4. **Test an audio file**:
    ```bash
    python3 -m predicto path/to/audio.wav
    ```

## License
This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
