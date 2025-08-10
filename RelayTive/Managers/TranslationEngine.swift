//
//  TranslationEngine.swift
//  RelayTive
//
//  Translation engine with real HuBERT CoreML model for speech embedding extraction
//

import Foundation
import CoreML
import SwiftUI
import AVFoundation
import Accelerate

@MainActor
class TranslationEngine: ObservableObject {
    @Published var isProcessing = false
    @Published var lastProcessingTime: TimeInterval = 0
    
    private var hubertModel: MLModel?
    private let audioEngine = AVAudioEngine()
    private let audioFormat = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
    
    // Cache to prevent identical embedding extractions - using hash keys to prevent memory leaks
    private var embeddingCache: [String: [Float]] = [:] 
    private let maxCacheSize = 20
    
    // Reused MLMultiArray buffer to prevent memory churn
    private var reusableInputBuffer: MLMultiArray?
    private let expectedInputSize = 480000
    
    // Debug diagnostics
    #if DEBUG
    var debugDiagnosticsEnabled = true
    private(set) var lastEmbedding: [Float]? = nil
    #else
    var debugDiagnosticsEnabled = false
    #endif
    
    init() {
        loadHuBERTModel()
    }
    
    // MARK: - Model Loading
    
    private func loadHuBERTModel() {
        print("Loading HuBERT CoreML model...")
        
        do {
            // Try loading the compiled model first (.mlmodelc)
            if let compiledModelURL = Bundle.main.url(forResource: "RelayTive_HuBERT", withExtension: "mlmodelc") {
                print("Found compiled model: RelayTive_HuBERT.mlmodelc")
                
                let config = MLModelConfiguration()
                config.computeUnits = .all // Use Neural Engine + GPU + CPU
                
                hubertModel = try MLModel(contentsOf: compiledModelURL, configuration: config)
                print("HuBERT model loaded successfully from compiled model (.mlmodelc)")
                print("Model input description: \(hubertModel?.modelDescription.inputDescriptionsByName ?? [:])")
                print("Model output description: \(hubertModel?.modelDescription.outputDescriptionsByName ?? [:])")
                return
            }
            
            // Fallback: Try original package format (.mlpackage)
            guard let modelURL = Bundle.main.url(forResource: "RelayTive_HuBERT", withExtension: "mlpackage") else {
                print("Error: Neither compiled (.mlmodelc) nor package (.mlpackage) model found in bundle")
                
                // Final attempt: direct path lookup for mlmodelc
                if let bundlePath = Bundle.main.resourcePath {
                    let directModelPath = "\(bundlePath)/RelayTive_HuBERT.mlmodelc"
                    print("Trying direct path for compiled model: \(directModelPath)")
                    
                    if FileManager.default.fileExists(atPath: directModelPath) {
                        print("Compiled model exists at direct path!")
                        let directURL = URL(fileURLWithPath: directModelPath)
                        
                        let config = MLModelConfiguration()
                        config.computeUnits = .all
                        
                        hubertModel = try MLModel(contentsOf: directURL, configuration: config)
                        print("HuBERT model loaded successfully via direct path (.mlmodelc)")
                        print("Model input description: \(hubertModel?.modelDescription.inputDescriptionsByName ?? [:])")
                        print("Model output description: \(hubertModel?.modelDescription.outputDescriptionsByName ?? [:])")
                        return
                    } else {
                        print("No model found at any expected location")
                    }
                }
                return
            }
            
            let config = MLModelConfiguration()
            config.computeUnits = .all // Use Neural Engine + GPU + CPU
            
            hubertModel = try MLModel(contentsOf: modelURL, configuration: config)
            print("HuBERT model loaded successfully")
            print("Model input description: \(hubertModel?.modelDescription.inputDescriptionsByName ?? [:])")
            print("Model output description: \(hubertModel?.modelDescription.outputDescriptionsByName ?? [:])")
            
        } catch {
            print("Failed to load HuBERT model: \(error)")
            hubertModel = nil
        }
    }
    
    // MARK: - Translation Interface
    
    func translateAudio(_ audioData: Data) async -> String? {
        isProcessing = true
        defer { isProcessing = false }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let translation = await performHuBERTInference(audioData)
        
        let endTime = CFAbsoluteTimeGetCurrent()
        lastProcessingTime = endTime - startTime
        
        print("HuBERT processing completed in \(String(format: "%.2f", lastProcessingTime)) seconds")
        return translation
    }
    
    // MARK: - Public Embedding Extraction API
    
    /// Extract embeddings from a file URL (for real recordings with headers)
    func extractEmbeddings(fromFile url: URL) async -> [Float]? {
        let fileHash = url.absoluteString
        
        // Check cache first
        if let cachedEmbeddings = embeddingCache[fileHash] {
            print("🎯 Using cached embeddings for file: \(url.lastPathComponent)")
            return cachedEmbeddings
        }
        
        isProcessing = true
        defer { isProcessing = false }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard hubertModel != nil else {
            print("HuBERT model not loaded")
            return nil
        }
        
        do {
            let file = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(file.length)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
                print("Failed to create buffer for file")
                return nil
            }
            try file.read(into: buffer)
            
            guard let embeddings = await extractEmbeddings(from: buffer) else {
                return nil
            }
            
            let endTime = CFAbsoluteTimeGetCurrent()
            lastProcessingTime = endTime - startTime
            
            // Cache the result
            embeddingCache[fileHash] = embeddings
            
            print("HuBERT file processing completed in \(String(format: "%.2f", lastProcessingTime)) seconds")
            return embeddings
        } catch {
            print("Failed to read audio file: \(error)")
            return nil
        }
    }
    
    /// Extract embeddings from a PCM buffer (for internal audio segments)
    func extractEmbeddings(from buffer: AVAudioPCMBuffer) async -> [Float]? {
        isProcessing = true
        defer { isProcessing = false }
        
        guard let model = hubertModel else {
            print("HuBERT model not loaded")
            return nil
        }
        
        // Use unified preprocessing function
        guard let processedBuffer = preprocessAudioBuffer(buffer, targetSR: 16000) else {
            print("Failed to preprocess audio buffer")
            return nil
        }
        
        // Extract embeddings directly from processed buffer
        return extractHuBERTEmbeddings(from: processedBuffer, using: model)
    }
    
    /// Legacy Data-based method - only for real files with headers
    func extractEmbeddings(_ audioData: Data) async -> [Float]? {
        // Check if this is a RIFF/WAVE file (real file with headers)
        if audioData.count >= 12 &&
           audioData[0] == 0x52 && audioData[1] == 0x49 && audioData[2] == 0x46 && audioData[3] == 0x46 && // "RIFF"
           audioData[8] == 0x57 && audioData[9] == 0x41 && audioData[10] == 0x56 && audioData[11] == 0x45 {   // "WAVE"
            // Use file-based method for WAV files
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
            do {
                try audioData.write(to: tempURL)
                defer { try? FileManager.default.removeItem(at: tempURL) }
                return await extractEmbeddings(fromFile: tempURL)
            } catch {
                print("Failed to write temp WAV file: \(error)")
                return nil
            }
        } else {
            // Construct buffer and use buffer path (no temp WAV round-trip)
            guard let buffer = createAudioBuffer(from: audioData) else {
                print("Failed to create buffer from raw audio data")
                return nil
            }
            return await extractEmbeddings(from: buffer)
        }
    }
    
    /// Cache sanity test to detect collisions
    private func checkCacheCollision(key: String, embeddings: [Float]) {
        if debugDiagnosticsEnabled, let cached = embeddingCache[key] {
            // Check if embeddings are suspiciously identical
            let allSame = zip(embeddings, cached).allSatisfy { abs($0 - $1) < 1e-6 }
            if allSame {
                print("[Diag] suspect cache key collision for \(key)")
            }
        }
    }
    
    // MARK: - HuBERT Processing Pipeline
    
    private func performHuBERTInference(_ audioData: Data) async -> String? {
        // Use the unified Data-based extraction path
        guard let embeddings = await extractEmbeddings(audioData) else {
            return nil
        }
        return mapEmbeddingsToTranslation(embeddings)
    }


    /// Single unified preprocessing function used by ALL callers
    private func preprocessAudioBuffer(_ inputBuffer: AVAudioPCMBuffer, targetSR: Double = 16000) -> AVAudioPCMBuffer? {
        do {
            // Step 1: Resample to target sample rate mono if needed
            let targetFormat = AVAudioFormat(standardFormatWithSampleRate: targetSR, channels: 1)!
            let resampledBuffer: AVAudioPCMBuffer
            
            if inputBuffer.format.sampleRate == targetSR && inputBuffer.format.channelCount == 1 {
                resampledBuffer = inputBuffer
            } else {
                resampledBuffer = try resampleAudio(buffer: inputBuffer, targetSampleRate: targetSR)
            }
            
            let inputSamples = Int(resampledBuffer.frameLength)
            let expectedSamples = 480000
            
            // Step 2: Create output buffer with exactly 480k samples
            guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(expectedSamples)) else {
                print("Failed to create output buffer")
                return nil
            }
            
            outputBuffer.frameLength = AVAudioFrameCount(expectedSamples)
            
            guard let inputPtr = resampledBuffer.floatChannelData?[0],
                  let outputPtr = outputBuffer.floatChannelData?[0] else {
                print("Failed to get channel data")
                return nil
            }
            
            // Step 3: Normalize to [-1, 1] range
            var maxValue: Float = 0
            for i in 0..<inputSamples {
                maxValue = max(maxValue, abs(inputPtr[i]))
            }
            let normalizationFactor = maxValue > 1e-6 ? 1.0 / maxValue : 1.0
            
            // Step 4: Copy normalized samples and zero-pad tail to 480k
            let copyCount = min(inputSamples, expectedSamples)
            for i in 0..<copyCount {
                outputPtr[i] = inputPtr[i] * normalizationFactor
            }
            
            // Zero-pad the rest (ALWAYS pad, never repeat, never reject)
            if copyCount < expectedSamples {
                for i in copyCount..<expectedSamples {
                    outputPtr[i] = 0.0
                }
            }
            
            // Unified logging
            let padCount = max(0, expectedSamples - inputSamples)
            print("Preprocessed audio: \(inputSamples) samples at 16kHz")
            if padCount > 0 {
                print("Audio padded from \(inputSamples) to \(expectedSamples) samples")
            }
            
            // Call diagnostics after preprocessing
            diag(outputBuffer, paddedFrom: inputSamples)
            
            return outputBuffer
            
        } catch {
            print("Audio preprocessing error: \(error)")
            return nil
        }
    }
    
    /// Buffer analysis for diagnostics
    private func analyzeBuffer(_ buffer: AVAudioPCMBuffer) -> (frames: Int, rms: Float, peak: Float, zeroRatio: Float) {
        guard let channelData = buffer.floatChannelData?[0] else {
            return (0, 0, 0, 1.0)
        }
        
        let frameCount = Int(buffer.frameLength)
        var sumSquares: Float = 0
        var peak: Float = 0
        var zeroCount = 0
        
        for i in 0..<frameCount {
            let sample = channelData[i]
            sumSquares += sample * sample
            peak = max(peak, abs(sample))
            if sample == 0.0 { zeroCount += 1 }
        }
        
        let rms = frameCount > 0 ? sqrt(sumSquares / Float(frameCount)) : 0
        let zeroRatio = frameCount > 0 ? Float(zeroCount) / Float(frameCount) : 0
        
        return (frameCount, rms, peak, zeroRatio)
    }
    
    private func diag(_ buf: AVAudioPCMBuffer, paddedFrom originalFrames: Int) {
        #if DEBUG
        guard let ch = buf.floatChannelData?[0] else { return }
        let n = Int(buf.frameLength)
        var sum: Float = 0
        var zeros = 0
        for i in 0..<n { let v = ch[i]; sum += v*v; if v == 0 { zeros += 1 } }
        let rms = n > 0 ? sqrt(sum / Float(n)) : 0
        let pad = max(0, n - originalFrames)
        print("[Diag] frames=\(n) rms=\(rms) pad=\(pad) zeros=\(zeros)")
        #endif
    }
    
    /// Detect degenerate embedding patterns for diagnostics
    private func detectEmbeddingDegeneracy(_ embeddings: [Float]) -> [String] {
        var flags: [String] = []
        
        guard !embeddings.isEmpty else {
            flags.append("empty")
            return flags
        }
        
        // Check for all zeros
        let allZeros = embeddings.allSatisfy { $0 == 0.0 }
        if allZeros {
            flags.append("zeros")
            return flags
        }
        
        // Check for NaN or infinite values
        let hasNaN = embeddings.contains { !$0.isFinite }
        if hasNaN {
            flags.append("nan_inf")
        }
        
        // Check for very low norm (near zero)
        let norm = sqrt(embeddings.map { $0 * $0 }.reduce(0, +))
        if norm < 1e-6 {
            flags.append("low_norm")
        }
        
        // Check for constant values
        let first = embeddings[0]
        let allSame = embeddings.allSatisfy { abs($0 - first) < 1e-6 }
        if allSame {
            flags.append("constant")
        }
        
        // Check for very high norm (potential overflow)
        if norm > 1e6 {
            flags.append("high_norm")
        }
        
        return flags
    }
    
    /// Extract HuBERT embeddings from preprocessed 16kHz buffer (should be exactly 480k samples)
    private func extractHuBERTEmbeddings(from buffer: AVAudioPCMBuffer, using model: MLModel) -> [Float]? {
        do {
            let expectedInputSize = 480000
            let frameCount = Int(buffer.frameLength)
            
            guard frameCount == expectedInputSize else {
                print("❌ Buffer has \(frameCount) samples, expected \(expectedInputSize)")
                return nil
            }
            
            guard let audioPtr = buffer.floatChannelData?[0] else {
                print("❌ Failed to get audio channel data")
                return nil
            }
            
            // Reuse or create MLMultiArray buffer
            if reusableInputBuffer == nil {
                reusableInputBuffer = try MLMultiArray(shape: [1, NSNumber(value: expectedInputSize)], dataType: .float32)
            }
            
            guard let inputArray = reusableInputBuffer else {
                print("Failed to get reusable buffer")
                return nil
            }
            
            // Copy audio data directly to MLMultiArray (no intermediate float array)
            for index in 0..<expectedInputSize {
                inputArray[index] = NSNumber(value: audioPtr[index])
            }
            
            // Create the input dictionary using the correct key from model description
            let input = try MLDictionaryFeatureProvider(dictionary: ["audio_input": inputArray])
            
            // Run inference
            let output = try model.prediction(from: input)
            
            // Extract embeddings using the correct output key from model description
            if let embeddingArray = output.featureValue(for: "embedding_output")?.multiArrayValue {
                let embeddings = (0..<embeddingArray.count).map { index in
                    embeddingArray[index].floatValue 
                }
                
                print("✅ Successfully extracted HuBERT embeddings: \(embeddings.count) dimensions")
                
                // Diagnostics for embeddings
                #if DEBUG
                let embNorm = sqrt(embeddings.map { $0 * $0 }.reduce(0, +))
                var dPrev: Float = -1
                
                if let lastEmb = lastEmbedding, lastEmb.count == embeddings.count {
                    let diff = zip(embeddings, lastEmb).map { $0 - $1 }
                    dPrev = sqrt(diff.map { $0 * $0 }.reduce(0, +))
                }
                
                print("[Diag] embNorm=\(String(format: "%.3f", embNorm)) dPrev=\(dPrev >= 0 ? String(format: "%.3f", dPrev) : "n/a")")
                lastEmbedding = embeddings
                #endif
                
                return embeddings
            } else {
                print("❌ No embeddings found in model output")
                print("Available output keys: \(Array(output.featureNames))")
                return nil
            }
            
        } catch {
            print("HuBERT inference error: \(error)")
            return nil
        }
    }
    
    private func mapEmbeddingsToTranslation(_ embeddings: [Float]) -> String? {
        // This is now a placeholder - the real mapping happens in the calling code
        // The TranslationEngine should not directly access DataManager
        // Instead, the calling code (TranslationView) should handle the lookup
        
        // For now, return nil to indicate no built-in mapping
        // The actual mapping will be done by DataManager.findTranslationForEmbeddings
        return nil
    }
    
    // MARK: - Model Information
    
    var modelInfo: String {
        let status = hubertModel != nil ? "Loaded" : "Not Loaded"
        let modelDescription = hubertModel?.modelDescription.metadata[.description] as? String ?? "N/A"
        
        return """
        HuBERT Model Status: \(status)
        Model Description: \(modelDescription)
        Processing Time: \(String(format: "%.2f", lastProcessingTime))s
        Compute Units: Neural Engine + GPU + CPU
        """
    }
    
    var isModelLoaded: Bool {
        return hubertModel != nil
    }
    
    // MARK: - Utilities
    
    func warmUpModel() {
        guard hubertModel != nil else {
            print("Cannot warm up: HuBERT model not loaded")
            return
        }
        
        Task {
            let fmt = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 16000)!
            buf.frameLength = 16000
            memset(buf.floatChannelData![0], 0, Int(buf.frameLength) * MemoryLayout<Float>.size)
            _ = await extractEmbeddings(from: buf)
        }
    }
    
    // MARK: - Audio Processing Helpers
    
    // This method is now deprecated - use buffer-based methods instead
    // Kept for backward compatibility only
    private func createAudioBuffer(from audioData: Data) -> AVAudioPCMBuffer? {
        print("⚠️ Using deprecated createAudioBuffer - prefer buffer-based methods")
        
        // Treat as headerless PCM (44.1kHz, 16-bit mono) from AVAudioRecorder
        let frameCount = audioData.count / 2 // 2 bytes per 16-bit sample
        let format = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            print("❌ Failed to create PCM buffer for raw audio")
            return nil
        }
        
        buffer.frameLength = AVAudioFrameCount(frameCount)
        
        // Copy raw PCM data to buffer
        audioData.withUnsafeBytes { bytes in
            let int16Ptr = bytes.bindMemory(to: Int16.self)
            let floatPtr = buffer.floatChannelData![0]
            
            // Convert Int16 to Float (normalization happens in preprocessing)
            for i in 0..<frameCount {
                floatPtr[i] = Float(int16Ptr[i]) / 32768.0
            }
        }
        
        return buffer
    }
    
    private func resampleAudio(buffer: AVAudioPCMBuffer, targetSampleRate: Double) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format
        let outputFormat = AVAudioFormat(standardFormatWithSampleRate: targetSampleRate, channels: inputFormat.channelCount)!
        
        // If already at target sample rate, return original buffer
        if inputFormat.sampleRate == targetSampleRate {
            return buffer
        }
        
        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw NSError(domain: "AudioProcessing", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create audio converter"])
        }
        
        let outputFrameCount = AVAudioFrameCount(Double(buffer.frameLength) * (targetSampleRate / inputFormat.sampleRate))
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            throw NSError(domain: "AudioProcessing", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create output buffer"])
        }
        
        var error: NSError?
        var inputConsumed = false
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if inputConsumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            inputConsumed = true
            outStatus.pointee = .haveData
            return buffer
        }
        
        if status == .error, let error = error {
            throw error
        }
        
        return outputBuffer
    }
    
    private func convertBufferToFloatArray(_ buffer: AVAudioPCMBuffer) -> [Float]? {
        guard let channelData = buffer.floatChannelData else { return nil }
        
        let frameLength = Int(buffer.frameLength)
        let floatArray = Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        
        return floatArray
    }
    
    private func normalizeAudio(_ audioArray: [Float]) -> [Float] {
        guard !audioArray.isEmpty else { return [] }
        
        // Find the maximum absolute value
        let maxValue = audioArray.map(abs).max() ?? 1.0
        
        // Avoid division by zero
        guard maxValue > 0 else { return audioArray }
        
        // Normalize to [-1, 1] range
        return audioArray.map { $0 / maxValue }
    }
    
    // MARK: - Embedding Cache Management
    
    private func manageCache(audioHash: String, embeddings: [Float]) {
        // Cache sanity check
        checkCacheCollision(key: audioHash, embeddings: embeddings)
        
        embeddingCache[audioHash] = embeddings
        
        // Limit cache size to prevent memory bloat
        if embeddingCache.count > maxCacheSize {
            // Remove oldest entries - hash keys are more memory-efficient
            let keysToRemove = Array(embeddingCache.keys.prefix(5))
            for key in keysToRemove {
                embeddingCache.removeValue(forKey: key)
            }
            print("🧹 Embedding cache cleaned, now has \(embeddingCache.count) entries")
        }
    }
}
