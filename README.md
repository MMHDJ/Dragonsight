# Dragonsight YOLO Inference

This repository contains a Python demonstration script to load a custom YOLO model and compute cosine similarity between two Clash of Clans defensive layout images, based on detected building positions around the Town Hall.

## Contents

- `main.py`: main script that loads the model, runs inference on two images, and computes similarity between object vectors.
- `requirements.txt`: required Python dependencies.
- `Dragonsight_1.1.pt`: custom YOLO model used for detection.
- `validation/`: folder containing validation images used by the script.

## Requirements

- Python 3.9+
- `pip`

## Installation

1. Open a terminal in the folder containing this README.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script from the same folder:

```bash
python main.py
```

The script loads the `Dragonsight_1.1.pt` model and performs detection on the following two images:

- `./validation/181_home.jpg`
- `./validation/181_war_scout_180.jpg`

It then calculates and prints the highest cosine similarity score between the generated feature vectors.

## How it works

The script:

1. Loads a YOLO Ultralytics model.
2. Detects objects in each image.
3. Finds the Town Hall location (`Town Hall`).
4. Builds orientation vectors for specific classes around the Town Hall.
5. Computes cosine similarity between vectors from the two images.

## Notes

- The script is designed as a demonstration and can be adapted for other image pairs.
- The current model version was trained only on TH17 layouts, so inaccuracies may occur with other Town Hall levels.

## License

This project is released under the MIT License.
