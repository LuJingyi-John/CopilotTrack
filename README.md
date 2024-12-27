# CopilotTrack

Real-time object tracking implementation based on Co-tracker.

## Quick Start

### Environment Setup
```bash
# Create and activate conda environment
conda create -n copilot python=3.8
conda activate copilot

# Install co-tracker
pip install git+https://github.com/facebookresearch/co-tracker.git

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Download Pre-trained Model
```bash
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
cd ..
```

## Requirements
See `requirements.txt` for detailed dependencies.