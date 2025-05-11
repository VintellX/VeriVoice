## Usage

### 1. Clone and Setup
```bash
git clone --recursive https://github.com/VintellX/VeriVoice.git
cd VeriVoice
```

### 2. Virtual Env Setup and Installing Dependencies
```bash
conda create -n VeriVoice python=3.10
conda activate VeriVoice
pip install -r requirements.txt
```

### 3. Prepare the dataset
- Follow the steps for DF-Audio-Dataset to create the dataset
- Dataset should be inside DF-Audio-Dataset/DeepFake
- Inside it, there should be two folders:
    - `RealAudios/` with real `.wav` files
    - `FakeAudios/` with fake `.wav` files

```bash
DeepFake/
├── FakeAudios/
│   ├── speaker1_recordings/
│   │   └── speaker1_output1.wav
│   │   └── ...
│   └── ...
└── RealAudios/
    ├── speaker1_recordings/
    │   └── speaker1_input1.wav
    │   └── ...
    └── ...
```

### 4. Train the Model
Run the jupyter notebook file.
```bash
python -m ipykernel install --user --name=VeriVoice --display-name "VeriVoice-Env"
jupyter notebook
```

### 5. Predict New Audio Files
Run the predicto.py file on your CLI:
```bash
python3 -m predicto path/to/audio.wav
```
