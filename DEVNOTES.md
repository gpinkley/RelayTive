# DEVNOTES — RelayTive Phonetic Pipeline (Phonetic-First w/ Fallbacks)

**Last updated:** 2025-08-12

This document describes the phonetic pipeline that now ships in the app: how audio turns into discrete **phonetic units** and ultimately into a **meaning** with a calibrated confidence, plus what’s configurable, what to demo, and what’s still TODO.

---

## What changed (at a glance)

* **Execution order:** We now try **Phonetic → Compositional → Traditional** (in that order).
  Phonetic classification is attempted first; if confidence is low, we fall back to compositional pattern matching; if that fails, we fall back to whole-utterance embedding matching.

* **Real-time VAD:** Streaming voice activity detection (VAD) uses **energy (dB)** + **spectral flux** (vDSP FFT). It gates frame processing and reduces noise-only work.

* **Per-frame embeddings:** Short frames are **tiled** to match the model’s 480k-sample input. No 480k padding footgun in app code.

* **Online clustering:** **OnlineKMeans** learns a **160-unit** user codebook on the fly (cosine space, adaptive EMA, L2-normalized centroids).

* **Classifier:** **Nearest-centroid** in cosine space → **temperature softmax** (T=10) → calibrated **confidence** + **margin** (top1−top2).
  Confidence is **fused**: `0.6 * embedding_score + 0.4 * phonetic_similarity`.

* **Debug:** A **Phonetic Debug View** shows live metrics (energy dB, spectral flux, active units, confidence) and has a lightweight **DTW** tool for phonetic strings.

---

## Acronyms (quick)

* **VAD:** Voice Activity Detection — detects speech vs. silence.
* **FFT:** Fast Fourier Transform — spectral analysis (Accelerate/vDSP).
* **DTW:** Dynamic Time Warping — sequence alignment similarity.
* **EMA:** Exponential Moving Average — smooth centroid updates.
* **HuBERT:** Self-supervised speech model used for embeddings (base LS-960).

---

## High-level architecture

```
Mic Audio
  ↓
VAD (energy dB + spectral flux, hangover)
  ↓ (voiced frames only)
Frame windowing (20 ms, hop 10 ms @ 16 kHz → 320/160 samples)
  ↓
Per-frame embedding (HuBERT; frame tiled to 480k for Core ML)
  ↓
OnlineKMeans.observe → unit IDs (0…159) per frame
  ↓
Collapse repeats → “U12 U7 U3 …” phonetic string
  ↓
NearestCentroidClassifier
  • cosine → softmax(T=10)
  • + phonetic DTW similarity (0.6/0.4 fusion)
  ↓
Meaning + confidence + margin (+needsConfirmation if low)
  ↓
Fallbacks (Compositional → Traditional) if confidence insufficient
```

---

## Components & key APIs

### 1) `VADProcessor` (Models/DSP)

* **Frames:** 20 ms frame (320 samples at 16 kHz); **10 ms hop** (160 samples).
* **Spectral flux:** vDSP FFT (`vDSP_fft_zrip`) + **positive differences** between consecutive magnitude spectra.
* **Energy (dB):** Windowed RMS → `20 * log10(rms)`.
* **States:** `.silence`, `.speechStart`, `.speech`, `.hangover`, `.speechEnd`.
* **Thresholding:** Dynamic flux threshold from a small history buffer; energy threshold ramps (e.g., −35 dB toward −40 dB).
* **Durations:**

  * `minSpeechFrames ≈ 10` → \~100 ms needed to enter speech.
  * `minSilenceFrames ≈ 5` → \~50 ms needed to exit to silence.
  * `hangoverMaxFrames ≈ 20` → \~200 ms to avoid chopping trailing phones.

> Output: voiced frame indices/times + per-frame metrics (energy, flux, state).

---

### 2) `TranslationEngine` (Models/Engines)

* **New helper:** `extractFrameEmbedding(from: AVAudioPCMBuffer) async -> [Float]?`

  * **Tiling:** Copies a short frame repeatedly to fill the model’s **expectedInputSize (≈480k samples)**.
  * Runs the Core ML HuBERT model and returns the embedding vector (dim **768** for base).

> Keep this name in docs/code—Cursor relies on it.

---

### 3) `OnlineKMeans` (Models/ML)

* **Config:** `k = 160`, `dim = 768`.
* **Space:** Cosine space with **L2 normalization** of centroids and samples.
* **Update:** **Adaptive EMA** — learning rate ≈ `initialLR / sqrt(n)` per cluster (initialLR ≈ 0.3).
* **API:**

  * `observe(_ vector: [Float]) -> Int` → returns assigned unit ID.
  * `getCentroids() -> [[Float]]`
  * `getClusterSizes() -> [Int]`

> Learns the user’s personal unit inventory incrementally, frame by frame.

---

### 4) `NearestCentroidClassifier` (Models/ML)

* **Inputs:** an utterance-level **embedding** and an optional **phonetic string** (e.g., `"U12 U7 U3"`).
* **Scoring:**

  1. **Cosine similarity** to each meaning centroid → **softmax** with **T=10**.
  2. **Phonetic similarity** via token-level DTW/Levenshtein in `[0, 1]`.
  3. **Fusion:** `final = 0.6 * embedProb + 0.4 * phoneticSim`.
* **Decision rules:**

  * `needsConfirmation` if `confidence < 0.70 || margin < 0.10`.
  * Returns `topMeaning`, `confidence`, `margin`, `alternatives` (top-2).
* **Learning:**

  * `updateWithExample(meaning:, embedding:, phoneticString:)` — EMA-update centroid; keep a bounded list of phonetic prototypes per meaning (e.g., last 10 strings).

---

### 5) `PhoneticTranscriptionEngine` (Models/Engines)

* **Pipeline:**
  VAD → framed audio → `TranslationEngine.extractFrameEmbedding` per voiced frame → `OnlineKMeans.observe` → **collapse repeats** → **unit string** (e.g., `"U12 U7 U3"`).
* **Return type:** `PhoneticForm`

  ```swift
  struct PhoneticForm: Codable {
      let unitIDs: [Int]           // e.g., [12, 7, 3]
      let unitString: String       // "U12 U7 U3"
      let canonicalUnitsString: String // same for now; reserved for future canonicalization
      let readableSpelling: String? // optional (TODO: unit→symbol map)
      let confidence: Float
      let timestamp: Date
  }
  ```
* **Notes:** The **human-readable phonetic alphabet** is **TODO** (see “Roadmap”).

---

### 6) `DataManager` integration (Models)

* **Init:**
  `initializePhoneticPipeline(with translationEngine:)` creates the phonetic engine & classifier.
* **Path order:**
  `findTranslationForAudio(...)` now does:
  **Phonetic classify → Compositional match → Traditional embed match**.
* **Feedback loop:**
  `confirmPhoneticTranslation(audioData:, meaning:, using:)` re-transcribes & updates the classifier with a confirmed example (embedding + phonetic string).
* **Compatibility:** Prior compositional and traditional paths are kept for fallbacks.

---

### 7) `PhoneticDebugView` (Views/Debug)

* Shows **live VAD metrics**, **active unit count**, **last confidence**, and the **current phonetic string** (monospaced).
* Includes a **local DTW tool** for comparing two phonetic strings.
* Uses `VADProcessor.VADState` for state display.

---

## Tunables (safe defaults)

* **VAD:**

  * `frame = 320`, `hop = 160` @ 16 kHz
  * `energyThresholdStart = −35 dB → −40 dB`
  * `fluxThreshold = mean(fluxHistory) + factor * std` (factor ≈ 2.5)
  * `minSpeechFrames = 10`, `minSilenceFrames = 5`, `hangoverMaxFrames = 20`
* **K-Means:**

  * `k = 160`, `initialLR ≈ 0.3`, decay `1/√n`, **L2 normalize** centroids
* **Classifier:**

  * `T = 10`, `pMin = 0.70`, `marginMin = 0.10`
  * Fusion weights: **0.6 embed / 0.4 phonetic**

---

## Performance & memory

* **Tiling** avoids huge zero-padding cost; per-frame inference stays sub-10 ms on recent iOS devices.
* **FFT setup** is created once and reused.
* **No excessive allocations** in the hot path (reuse buffers).
* **DataManager** prunes to **\~15 recent** training examples in memory to reduce churn.

Concurrency (Swift 6):

* `@preconcurrency import AVFoundation` to silence non-annotated Sendable warnings.
* `PhoneticTranscriptionEngine` marked `@unchecked Sendable`; all work is serialized internally.
* `withCheckedContinuation { queue.async { Task { … await … } } }` for async boundaries.

---

## Demo script (short)

1. **Record or stream audio** on the Translate tab.
2. **Watch VAD meter** light up; pipeline extracts frame embeddings.
3. **Units appear** (collapsed) as `"U… U… U…"`.
4. **Classifier outputs** a meaning with confidence & margin.

   * If `needsConfirmation`, show the fallback or request caregiver confirmation.
5. **Feedback** → mark the correct meaning → classifier improves instantly (EMA).
6. Optional: Open **Phonetic Debug** to show metrics and DTW of two phonetic strings.

---

## Roadmap / TODO

* **Readable phonetic alphabet:** Add `unitToSymbol: [Int:String]` and render `readableSpelling` (and optional IPA).
* **Codebook persistence:** Save/restore OnlineKMeans centroids per user profile.
* **Embedding LRU cache:** Optional; useful for repeated previews.
* **Batch frame path:** An optional `extractFrameLevelEmbeddings` for HuBERT variants that return hidden state sequences.
* **Calibration tooling:** Confusion matrices by meaning, unit usage histograms, timing-sensitivity plots.

---

## Troubleshooting (top issues)

* **“Value of optional UnsafePointer must be unwrapped”**
  Ensure `buffer.floatChannelData?[0]` is safely unwrapped before use.

* **“Cannot find vDSP\_DFT\_FORWARD”**
  Use **`vDSP_fft_zrip`** (FFT) instead of DFT symbols on platforms where DFT isn’t available by default.

* **All-zeros frames or mostly zeros**
  Likely a format/stride issue when building `AVAudioPCMBuffer` or a channel that isn’t mono. Confirm 16 kHz, mono, float32.

* **No voiced segments**
  Reduce `energyThresholdStart` (e.g., from −35 dB to −40 dB) or lower `fluxThresholdFactor`.

---

## Licensing & attributions (what this build uses)

* **App & Library Code**
  © 2025 **Auteuristic Systems, Inc.** All rights reserved. Unless otherwise noted.

* **Apple Frameworks**
  AVFoundation, Accelerate/vDSP — under Apple Developer Program License.

* **Speech Embedding Model (HuBERT base LS-960)**

  * Checkpoint: `hubert_base_ls960.pt`
  * **SHA-256:** `1703cf8d2cdc76f8c046f5f6a9bcd224e0e6caf4744cad1a1f4199c32cac8c8d`
  * **Size:** `1,136,468,879` bytes
  * **License:** **Apache License 2.0** (model weights)
  * Paper: Hsu et al., 2021 — include citation in README or About.

> If you swap to a different checkpoint (e.g., CC BY-NC or other), update the LICENSE section and enforce “demo/research only” if needed.

---

## Files touched (overview)

* **Models/DSP**
  `VADProcessor.swift`
* **Models/ML**
  `OnlineKMeans.swift`, `NearestCentroidClassifier.swift`
* **Models/Engines**
  `PhoneticTranscriptionEngine.swift`, `TranslationEngine.swift` (tiled frame helper)
* **Models**
  `DataManager.swift` (phonetic-first + fallbacks), `TrainingExample.swift` (adds `PhoneticForm`)
* **Views/Debug**
  `PhoneticDebugView.swift` (live metrics, DTW)

---

## Contact

Questions / licensing / press: **Auteuristic Systems, Inc.**
Internal point: add phonetic symbol mapping (`unitToSymbol`) next, then persist centroids to the user profile.
