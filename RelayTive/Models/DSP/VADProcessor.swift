//
//  VADProcessor.swift
//  RelayTive
//
//  Streaming Voice Activity Detection with energy and spectral flux
//

import Foundation
import Accelerate
import AVFoundation

/// Voice Activity Detection processor using energy and spectral flux analysis
class VADProcessor {
    
    // MARK: - Configuration
    private let frameSize: Int
    private let hopSize: Int
    private let sampleRate: Double
    private let windowFunction: [Float]
    
    // Energy thresholds (dB)
    private let energyThresholdStart: Float = -35.0  // Start speech detection
    private let energyThresholdEnd: Float = -40.0    // End speech detection
    
    // Spectral flux parameters
    private var fluxHistory: [Float] = []
    private var fluxThreshold: Float = 0.0
    private let fluxHistorySize = 20
    private let fluxThresholdFactor: Float = 2.5
    
    // State machine
    enum VADState {
        case silence
        case speechStart
        case speech
        case speechEnd
        case hangover
    }
    
    private var currentState: VADState = .silence
    private var speechStartTime: TimeInterval = 0
    private var lastSpeechTime: TimeInterval = 0
    private var hangoverFrames = 0
    
    // Timing constraints (in frames)
    private let minSpeechFrames: Int
    private let minSilenceFrames: Int
    private let hangoverMaxFrames: Int
    
    // FFT setup
    private let fftSetup: vDSP_DFT_Setup
    private let fftSize: Int
    private var fftInputReal: [Float]
    private var fftInputImag: [Float]
    private var fftOutputReal: [Float]
    private var fftOutputImag: [Float]
    private var magnitudeSpectrum: [Float]
    private var previousMagnitude: [Float]
    
    // Frame timing
    private var frameIndex: Int = 0
    private let frameTimeStep: TimeInterval
    
    init(frameSize: Int = 320, hopSize: Int = 160, sampleRate: Double = 16000) {
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.sampleRate = sampleRate
        self.frameTimeStep = Double(hopSize) / sampleRate
        
        // Calculate timing constraints in frames
        self.minSpeechFrames = Int(0.1 * sampleRate / Double(hopSize))  // 100ms minimum speech
        self.minSilenceFrames = Int(0.05 * sampleRate / Double(hopSize)) // 50ms minimum silence
        self.hangoverMaxFrames = Int(0.2 * sampleRate / Double(hopSize))  // 200ms hangover
        
        // FFT size (next power of 2)
        self.fftSize = 1 << Int(ceil(log2(Double(frameSize))))
        
        // Create Hann window
        self.windowFunction = (0..<frameSize).map { i in
            let factor = 2.0 * Float.pi * Float(i) / Float(frameSize - 1)
            return 0.5 * (1.0 - cos(factor))
        }
        
        // Initialize FFT setup (real→complex)
        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(fftSize), vDSP_DFT_Direction.FORWARD) else {
            fatalError("Failed to create FFT setup")
        }
        self.fftSetup = setup
        
        // Initialize FFT buffers (real→complex)
        self.fftInputReal = Array(repeating: 0.0, count: fftSize)          // N
        self.fftInputImag = Array(repeating: 0.0, count: fftSize)          // N (zeros for zrop)
        self.fftOutputReal = Array(repeating: 0.0, count: fftSize / 2)     // N/2
        self.fftOutputImag = Array(repeating: 0.0, count: fftSize / 2)     // N/2
        self.magnitudeSpectrum = Array(repeating: 0.0, count: fftSize / 2) // N/2
        self.previousMagnitude = Array(repeating: 0.0, count: fftSize / 2) // N/2
    }
    
    deinit {
        vDSP_DFT_DestroySetup(fftSetup)
    }
    
    // MARK: - Public Interface
    
    /// Process a frame of audio and return VAD decision
    func processFrame(_ audioFrame: [Float]) -> VADResult {
        guard audioFrame.count >= frameSize else {
            return VADResult(
                isSpeech: false,
                energy: -80.0,
                spectralFlux: 0.0,
                state: currentState,
                timestamp: getCurrentTime()
            )
        }
        
        let currentTime = getCurrentTime()
        
        // Calculate frame energy in dB
        let energy = calculateFrameEnergyDB(audioFrame)
        
        // Calculate spectral flux
        let flux = calculateSpectralFlux(audioFrame)
        updateFluxThreshold(flux)
        
        // State machine logic
        let previousState = currentState
        updateState(energy: energy, flux: flux, timestamp: currentTime)
        
        let isSpeech = (currentState == .speech || currentState == .speechStart || currentState == .hangover)
        
        frameIndex += 1
        
        return VADResult(
            isSpeech: isSpeech,
            energy: energy,
            spectralFlux: flux,
            state: currentState,
            timestamp: currentTime,
            speechStart: (previousState != .speechStart && currentState == .speechStart) ? currentTime : nil,
            speechEnd: (previousState == .hangover && currentState == .silence) ? currentTime : nil
        )
    }
    
    /// Process entire audio buffer and return segments
    func processBuffer(_ buffer: AVAudioPCMBuffer) -> [VoicedSegment] {
        guard let channelData = buffer.floatChannelData?[0] else {
            return []
        }
        
        let frameCount = Int(buffer.frameLength)
        var segments: [VoicedSegment] = []
        var currentSegmentStart: TimeInterval?
        
        // Process overlapping frames
        var frameStart = 0
        while frameStart + frameSize <= frameCount {
            let frame = Array(UnsafeBufferPointer(start: channelData + frameStart, count: frameSize))
            let result = processFrame(frame)
            
            // Track speech segments
            if let speechStart = result.speechStart {
                currentSegmentStart = speechStart
            }
            
            if let speechEnd = result.speechEnd, let segmentStart = currentSegmentStart {
                let segment = VoicedSegment(
                    startTime: segmentStart,
                    endTime: speechEnd,
                    confidence: 1.0
                )
                segments.append(segment)
                currentSegmentStart = nil
            }
            
            frameStart += hopSize
        }
        
        // Handle segment that might still be active at buffer end
        if let segmentStart = currentSegmentStart, currentState != .silence {
            let bufferDuration = Double(frameCount) / sampleRate
            let segment = VoicedSegment(
                startTime: segmentStart,
                endTime: bufferDuration,
                confidence: 0.8  // Lower confidence for incomplete segment
            )
            segments.append(segment)
        }
        
        return segments
    }
    
    /// Reset VAD state
    func reset() {
        currentState = .silence
        speechStartTime = 0
        lastSpeechTime = 0
        hangoverFrames = 0
        frameIndex = 0
        fluxHistory.removeAll()
        fluxThreshold = 0.0
        previousMagnitude = Array(repeating: 0.0, count: fftSize / 2)
    }
    
    // MARK: - Private Methods
    
    private func getCurrentTime() -> TimeInterval {
        return Double(frameIndex) * frameTimeStep
    }
    
    private func calculateFrameEnergyDB(_ frame: [Float]) -> Float {
        // Apply window function
        var windowedFrame = [Float](repeating: 0.0, count: frameSize)
        vDSP_vmul(frame, 1, windowFunction, 1, &windowedFrame, 1, vDSP_Length(frameSize))
        
        // Calculate RMS energy
        var sumSquares: Float = 0.0
        vDSP_svesq(windowedFrame, 1, &sumSquares, vDSP_Length(frameSize))
        
        let rms = sqrt(sumSquares / Float(frameSize))
        
        // Convert to dB with floor
        let energyDB = rms > 1e-10 ? 20.0 * log10(rms) : -100.0
        return energyDB
    }
    
    private func calculateSpectralFlux(_ frame: [Float]) -> Float {
        // Apply window function
        var windowedFrame = [Float](repeating: 0.0, count: frameSize)
        vDSP_vmul(frame, 1, windowFunction, 1, &windowedFrame, 1, vDSP_Length(frameSize))
        
        // Zero-pad for FFT
        fftInputReal[0..<frameSize] = ArraySlice(windowedFrame)
        for i in frameSize..<fftSize {
            fftInputReal[i] = 0.0
        }
        
        // Keep imaginary input at zero each call
        for i in 0..<fftSize {
            fftInputImag[i] = 0.0
        }
        
        // Perform DFT (real→complex). zrop requires non-nil real AND imag inputs.
        // imag may be all zeros but must be a valid pointer.
        fftInputReal.withUnsafeBufferPointer { inReal in
            fftInputImag.withUnsafeBufferPointer { inImag in
                fftOutputReal.withUnsafeMutableBufferPointer { outReal in
                    fftOutputImag.withUnsafeMutableBufferPointer { outImag in
                        vDSP_DFT_Execute(
                            fftSetup,
                            inReal.baseAddress!,    // real input (N)
                            inImag.baseAddress!,    // imag input (N) – zeros OK
                            outReal.baseAddress!,   // real output (N/2)
                            outImag.baseAddress!    // imag output (N/2)
                        )
                    }
                }
            }
        }
        
        // Calculate magnitude spectrum
        let halfSize = fftSize / 2
        for i in 0..<halfSize {
            let real = fftOutputReal[i]
            let imag = fftOutputImag[i]
            magnitudeSpectrum[i] = sqrt(real * real + imag * imag)
        }
        
        // Calculate spectral flux (positive differences only)
        var flux: Float = 0.0
        for i in 0..<halfSize {
            let diff = magnitudeSpectrum[i] - previousMagnitude[i]
            if diff > 0 {
                flux += diff
            }
        }
        
        // Update previous magnitude
        previousMagnitude = magnitudeSpectrum
        
        return flux
    }
    
    private func updateFluxThreshold(_ flux: Float) {
        fluxHistory.append(flux)
        
        if fluxHistory.count > fluxHistorySize {
            fluxHistory.removeFirst()
        }
        
        if fluxHistory.count >= 5 {
            // Calculate median and set threshold
            let sortedFlux = fluxHistory.sorted()
            let median = sortedFlux[sortedFlux.count / 2]
            fluxThreshold = median * fluxThresholdFactor
        }
    }
    
    private func updateState(energy: Float, flux: Float, timestamp: TimeInterval) {
        let energyActive = energy > energyThresholdStart
        let energyInactive = energy < energyThresholdEnd
        let fluxActive = flux > fluxThreshold && fluxThreshold > 0
        
        switch currentState {
        case .silence:
            if energyActive || fluxActive {
                currentState = .speechStart
                speechStartTime = timestamp
                lastSpeechTime = timestamp
                hangoverFrames = 0
            }
            
        case .speechStart:
            lastSpeechTime = timestamp
            hangoverFrames = 0
            
            let framesSinceSpeechStart = Int((timestamp - speechStartTime) / frameTimeStep)
            if framesSinceSpeechStart >= minSpeechFrames {
                currentState = .speech
            } else if energyInactive && !fluxActive {
                // False start - go back to silence
                currentState = .silence
            }
            
        case .speech:
            if energyActive || fluxActive {
                lastSpeechTime = timestamp
                hangoverFrames = 0
            } else {
                // Start hangover period
                currentState = .hangover
                hangoverFrames = 1
            }
            
        case .speechEnd:
            // Transition state - move to hangover or silence
            currentState = .hangover
            hangoverFrames = 1
            
        case .hangover:
            if energyActive || fluxActive {
                // Speech resumed - go back to speech
                currentState = .speech
                lastSpeechTime = timestamp
                hangoverFrames = 0
            } else {
                hangoverFrames += 1
                if hangoverFrames >= hangoverMaxFrames {
                    currentState = .silence
                    hangoverFrames = 0
                }
            }
        }
    }
}

// MARK: - Support Structures

struct VADResult {
    let isSpeech: Bool
    let energy: Float           // dB
    let spectralFlux: Float
    let state: VADProcessor.VADState
    let timestamp: TimeInterval
    let speechStart: TimeInterval?  // Non-nil when speech starts
    let speechEnd: TimeInterval?    // Non-nil when speech ends
    
    init(isSpeech: Bool, energy: Float, spectralFlux: Float, state: VADProcessor.VADState, timestamp: TimeInterval, speechStart: TimeInterval? = nil, speechEnd: TimeInterval? = nil) {
        self.isSpeech = isSpeech
        self.energy = energy
        self.spectralFlux = spectralFlux
        self.state = state
        self.timestamp = timestamp
        self.speechStart = speechStart
        self.speechEnd = speechEnd
    }
}

struct VoicedSegment {
    let startTime: TimeInterval
    let endTime: TimeInterval
    let confidence: Float
    
    var duration: TimeInterval {
        return endTime - startTime
    }
    
    var isValid: Bool {
        return duration > 0.05 && confidence > 0.5  // At least 50ms and reasonable confidence
    }
}