# Project: RelayTive
RelayTive is an iOS assistive app that translates atypical vocalizations into standardized speech via caregiver-provided examples.

## Current Architecture
- CoreML HuBERT (frozen)
- Whole-utterance nearest-neighbor matching
- No sub-pattern detection or compositional generalization yet

## Constraints
- iPhone 16e target, on-device only, privacy-first
- Quantized/distilled models allowed
- Continual on-device learning required

## Research Insights
- [Condensed research summary here]

## Claude Code Usage Guidelines
- Always make a plan first — do not code until plan approved
- Compare HuBERT, Wav2Vec2/WavLM, Whisper, MMS for suitability
- Recommend best architecture with resource constraints in mind
- When coding, prefer modular Swift implementations
- Keep suggestions CoreML-compatible
