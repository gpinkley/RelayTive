RelayTive is a privacy-first iOS app that enables live translation of atypical (nonstandard or neurodivergent) speech patterns into clear, typical language—entirely on-device.
It uses Apple’s CoreML and the HuBERT model for secure, customizable, real-time communication support, especially for nonverbal or speech-atypical individuals and their caregivers.

🚀 Features
Three-tab SwiftUI interface:

Translation — Real-time speech translation for atypical-to-typical language

Training — Record new speech samples and explanations to expand personalized vocabulary

Examples — Review, edit, and manage all trained utterances and their mapped meanings

On-device HuBERT CoreML integration:

No internet required; all processing happens locally for privacy

Neural Engine accelerated for real-time, battery-efficient translation

Customizable Training:

Caregivers can add new speech patterns (audio + literal explanation) to teach the app unique utterances

All mappings are user-created and editable; no predefined or canned examples

Live Speech Recognition:

Real iOS speech recognition for explanations

Manual entry supported as fallback

Accessibility:

Clean, guided workflow for non-technical users

No data ever leaves the device

🛠️ Architecture
Swift 5+, modular, pure SwiftUI app structure

CoreML model: HuBERT (RelayTive_HuBERT.mlpackage) for robust audio embedding extraction

Managers:

AudioManager (audio recording/playback)

DataManager (persistent utterance mapping)

TranslationEngine (HuBERT inference and translation logic)

SpeechRecognitionManager (iOS speech-to-text for explanations)

🗂️ Project Structure
Copy
Edit
RelayTive/
├── RelayTive.xcodeproj/
├── RelayTive/
│   ├── Assets.xcassets/
│   ├── ContentView.swift
│   ├── Managers/
│   ├── Models/
│   ├── Views/
│   └── RelayTiveApp.swift
├── RelayTiveTests/
├── RelayTiveUITests/
├── .gitignore
├── README.md
⚡ Getting Started
Prerequisites
Xcode 15+ (macOS)

iOS 17+ device for full speech recognition (simulator support limited)

Setup
Clone the repo:

sh
Copy
Edit
git clone <your_repo_url>
cd relaytive
Open the project in Xcode:

Double-click RelayTive.xcodeproj

Add HuBERT model:

The model file RelayTive_HuBERT.mlpackage must be added to RelayTive/Models/
(The model is not tracked in Git due to size/licensing; see .gitignore.)

Set privacy permissions:

Open Info.plist

Add:

NSMicrophoneUsageDescription = “RelayTive needs microphone access to translate speech”

NSSpeechRecognitionUsageDescription = “RelayTive uses speech recognition for caregiver explanations”

Build and run on device.

💡 How It Works
Training:

Record an atypical utterance (user’s unique vocalization).

Record or type the literal typical-language explanation.

Save the mapping; review and edit in Examples.

Translation:

Record new utterance in Translation tab.

If a match (or near-match) exists, the mapped meaning is displayed in real-time.

All data and mappings are local and user-controlled.

❗ Notes
No predefined mappings: All examples come from the user. The app starts empty.

Model input/output keys: See Model_Key_Update_Guide.md if your HuBERT model uses nonstandard keys.

Troubleshooting: See Xcode_Integration_Checklist.md and Cache_And_Metal_Issues.md.

📄 License
Proprietary. All rights reserved.
Contact the project owner for commercial or academic use.

🙋 Support
For bugs, feature requests, or support, please open an issue or contact the maintainer.
