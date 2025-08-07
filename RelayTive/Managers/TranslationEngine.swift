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
    
    // MARK: - HuBERT Processing Pipeline
    
    private func performHuBERTInference(_ audioData: Data) async -> String? {
        guard let model = hubertModel else {
            print("HuBERT model not loaded")
            return nil
        }
        
        // Step 1: Convert audio data to the format expected by HuBERT
        guard let audioArray = preprocessAudioForHuBERT(audioData) else {
            print("Failed to preprocess audio for HuBERT")
            return nil
        }
        
        // Step 2: Extract HuBERT embeddings using CoreML
        guard let embeddings = extractHuBERTEmbeddings(audioArray, using: model) else {
            print("Failed to extract HuBERT embeddings")
            return nil
        }
        
        // Step 3: Map embeddings to meaningful translation
        let translation = mapEmbeddingsToTranslation(embeddings)
        
        return translation
    }
    
    private func preprocessAudioForHuBERT(_ audioData: Data) -> [Float]? {
        guard !audioData.isEmpty else {
            print("Empty audio data")
            return nil
        }
        
        do {
            // Convert Data to AVAudioPCMBuffer
            guard let audioBuffer = createAudioBuffer(from: audioData) else {
                print("Failed to create audio buffer")
                return nil
            }
            
            // Resample to 16kHz if needed (HuBERT expects 16kHz)
            let resampledBuffer = try resampleAudio(buffer: audioBuffer, targetSampleRate: 16000)
            
            // Convert to Float array and normalize
            guard let floatArray = convertBufferToFloatArray(resampledBuffer) else {
                print("Failed to convert buffer to float array")
                return nil
            }
            
            // Normalize audio to [-1, 1] range
            let normalizedArray = normalizeAudio(floatArray)
            
            print("Preprocessed audio: \(normalizedArray.count) samples at 16kHz")
            return normalizedArray
            
        } catch {
            print("Audio preprocessing error: \(error)")
            return nil
        }
    }
    
    private func extractHuBERTEmbeddings(_ audioArray: [Float], using model: MLModel) -> [Float]? {
        do {
            // HuBERT model expects fixed input size: 480000 samples (30 seconds at 16kHz)
            let expectedInputSize = 480000
            let processedAudio: [Float]
            
            if audioArray.count > expectedInputSize {
                // Truncate if too long
                processedAudio = Array(audioArray.prefix(expectedInputSize))
                print("Audio truncated from \(audioArray.count) to \(expectedInputSize) samples")
            } else if audioArray.count < expectedInputSize {
                // Pad with zeros if too short
                processedAudio = audioArray + Array(repeating: 0.0, count: expectedInputSize - audioArray.count)
                print("Audio padded from \(audioArray.count) to \(expectedInputSize) samples")
            } else {
                // Perfect size
                processedAudio = audioArray
            }
            
            // Create MLMultiArray with the correct fixed size
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: expectedInputSize)], dataType: .float32)
            
            for (index, value) in processedAudio.enumerated() {
                inputArray[index] = NSNumber(value: value)
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
                print("Embedding sample (first 5 values): \(Array(embeddings.prefix(5)))")
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
        // For now, we'll use a simple clustering approach based on embedding patterns
        // In a full implementation, you'd have:
        // 1. A database of known embeddings paired with translations
        // 2. Similarity search (cosine distance, etc.)
        // 3. Clustering/classification of embedding patterns
        // 4. Contextual mapping based on user history
        
        guard !embeddings.isEmpty else { return nil }
        
        // Calculate some basic features from the embedding
        let embeddingMagnitude = sqrt(embeddings.map { $0 * $0 }.reduce(0, +))
        let embeddingMean = embeddings.reduce(0, +) / Float(embeddings.count)
        let embeddingVariance = embeddings.map { pow($0 - embeddingMean, 2) }.reduce(0, +) / Float(embeddings.count)
        
        print("Embedding features - Magnitude: \(embeddingMagnitude), Mean: \(embeddingMean), Variance: \(embeddingVariance)")
        
        // Simple rule-based mapping based on embedding characteristics
        // This is a placeholder - real implementation would use trained classifiers
        let translations: [String]
        
        if embeddingMagnitude > 10.0 {
            translations = ["Help me please", "I need assistance", "Something is wrong"]
        } else if embeddingVariance > 0.5 {
            translations = ["I want some water", "I'm hungry", "I need to go outside"]
        } else if embeddingMean > 0 {
            translations = ["Thank you", "I love you", "That feels good"]
        } else {
            translations = ["I'm tired", "It hurts here", "I don't feel well"]
        }
        
        let selectedTranslation = translations.randomElement() ?? "Unknown vocalization"
        print("Mapped to translation: \(selectedTranslation)")
        
        return selectedTranslation
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
            // Create dummy audio data (1 second of silence at 16kHz)
            let dummyAudioSamples = Array(repeating: Float(0.0), count: 16000)
            let dummyAudioData = Data(bytes: dummyAudioSamples, count: dummyAudioSamples.count * MemoryLayout<Float>.size)
            
            print("Warming up HuBERT model with dummy audio...")
            _ = await translateAudio(dummyAudioData)
            print("HuBERT model warmed up successfully")
        }
    }
    
    // MARK: - Audio Processing Helpers
    
    private func createAudioBuffer(from audioData: Data) -> AVAudioPCMBuffer? {
        // Convert WAV data to AVAudioPCMBuffer
        // Note: This assumes the recorded audio is in the correct format
        // You might need to adjust this based on your AudioManager's output format
        
        let bytesPerSample = 2 // 16-bit audio
        let sampleCount = audioData.count / bytesPerSample
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(sampleCount)) else {
            return nil
        }
        
        buffer.frameLength = AVAudioFrameCount(sampleCount)
        
        // Copy audio data to buffer
        audioData.withUnsafeBytes { bytes in
            let int16Pointer = bytes.bindMemory(to: Int16.self)
            if let channelData = buffer.int16ChannelData {
                channelData[0].update(from: int16Pointer.baseAddress!, count: sampleCount)
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
        let status = converter.convert(to: outputBuffer, error: &error) { _, _ in
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
}