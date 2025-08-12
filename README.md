# RelayTive

RelayTive is a privacy-first iOS app for translating atypical speech into clear, typical language.  
All processing is local using CoreML and HuBERT for on-device audio recognition.

## Key Features

- Train custom mappings between unique vocalizations and their explanations.
- Real-time translation using user-created training data.
- All data and processing stay on your device.

## Installation

1. Clone this repo and open `RelayTive.xcodeproj` in Xcode 15 or newer.
2. Add your HuBERT CoreML model (`RelayTive_HuBERT.mlpackage`) to `RelayTive/Models/`.
3. Set up microphone and speech recognition permissions in `Info.plist`.
4. Build and run on a physical iOS device.

## Privacy

RelayTive does not collect or transmit user data.  
All recognition and storage are performed locally.

## Note

The HuBERT model file is not included in this repository.

### Model provenance

We use HuBERT Base (LS-960) for frame-level embeddings.

- Checkpoint: `hubert_base_ls960.pt`
- SHA-256: `1703cf8d2cdc76f8c046f5f6a9bcd224e0e6caf4744cad1a1f4199c32cac8c8d`
- Size: 1,136,468,879 bytes
- Source: <URL>
- License (weights): <e.g., CC BY-NC 4.0> — demo/research only if non-commercial.
