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
