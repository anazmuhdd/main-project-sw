# TTS Configuration Utility

This folder contains tools to configure and optimize `pyttsx3` for the Radxa Cubie A7Z board.

## Requirements

If you are on a Linux-based system (like Radxa boards), you MUST install the following packages for `pyttsx3` to work correctly:

```bash
sudo apt update
sudo apt install espeak-ng libespeak1
```

Also, ensure you have the `pyttsx3` Python library:

```bash
pip install pyttsx3
```

## Usage

### 1. List Available Voices
Find out which voices are installed on your system.

```bash
python configure_tts.py --list
```

### 2. Test TTS with Specific Settings
Adjust the `rate`, `volume`, and `voice` to find the best quality.

```bash
python configure_tts.py --test "Hello, this is a test of the text to speech system." --voice 0 --rate 125 --volume 1.0
```

### 3. Save as Audio File
Useful for checking quality without needing a speaker connected during development.

```bash
python configure_tts.py --test "Hello world" --save output.mp3
```

## Tips for Better Quality on ARM Boards
- **Rate**: Lower values (e.g., 125-150) often sound more natural on espeak-ng.
- **Volume**: Keeping it at 1.0 might cause clipping on some low-power amplifiers; try 0.8 if distorted.
- **Voices**: Espeak-ng provides many variants. Use `--list` to see if `english-mbrola` or others are available for better naturalness.
