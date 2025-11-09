import os

GFPGAN_PATH = "models/GFPGANv1.4.pth"
REALSRC_PATH = "models/RealESRGAN_x4plus.pth"

# Check if files exist
if not os.path.exists(GFPGAN_PATH):
    raise FileNotFoundError(f"{GFPGAN_PATH} not found in Docker image!")

if not os.path.exists(REALSRC_PATH):
    raise FileNotFoundError(f"{REALSRC_PATH} not found in Docker image!")

# Example: load models
# gfpgan_model = GFPGANer(model_path=GFPGAN_PATH, ...)
# realesrgan_model = RRDBNet(model_path=REALSRC_PATH, ...)
