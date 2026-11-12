# DeiT-Based Multi-Fracture Detection

This project uses DeiT image features with classical classifiers to classify
fracture images. It also includes a Tkinter interface and an optional Telegram
bot with Gemini image analysis.

## Project contents

- `Main_Telegram.py` - application entry point
- `figs.py` - FIGS classifier implementation
- `Bone Break Classification/` - fracture image dataset
- `TestImages/` - sample images
- `model/` - trained classifiers and extracted feature data
- `background.png` - application background

The included images and model files are provided for this project. Check the
applicable dataset and image licenses before reusing or redistributing them.

## Setup

Install the Python packages used by the application: PyTorch, torchvision,
timm, NumPy, pandas, OpenCV, Matplotlib, Pillow, scikit-learn, joblib,
seaborn, requests, and python-telegram-bot.

Set the required credentials in your environment before using the related
features:

- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `GEMINI_API_KEY` - Google Gemini API key for image analysis

Do not commit credentials or `.env` files.

Run the application with Python:

```powershell
python Main_Telegram.py
```

This software is an experimental classification project and is not a medical
diagnostic tool.
