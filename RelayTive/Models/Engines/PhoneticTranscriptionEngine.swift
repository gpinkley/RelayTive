//
//  PhoneticTranscriptionEngine.swift
//  RelayTive
//
//  Complete phonetic transcription pipeline: VAD → frame embeddings → k-means → unit strings
//

import Foundation
@preconcurrency import AVFoundation
import Accelerate

/// Complete pipeline for phonetic transcription from audio to discrete unit strings
final class PhoneticTranscriptionEngine: @unchecked Sendable {
    
    // MARK: - Configuration
    private let frameSize: Int = 320        // 20ms at 16kHz
    private let hopSize: Int = 160          // 10ms hop
    private let sampleRate: Double = 16000
    private let windowFunction: [Float]
    
    // MARK: - Pipeline Components
    private let vadProcessor: VADProcessor
    private let onlineKMeans: OnlineKMeans
    private weak var translationEngine: TranslationEngine?
    
    // Optional symbol mapping
    private var unitToSymbolMap: [Int: String] = [:]
    
    // Threading
    private let queue = DispatchQueue(label: "com.relayTive.phoneticEngine", qos: .userInitiated)
    
    // MARK: - Initialization
    
    init(translationEngine: TranslationEngine, embeddingDim: Int = 768, k: Int = 160) {
        self.translationEngine = translationEngine
        
        // Initialize VAD
        self.vadProcessor = VADProcessor(
            frameSize: frameSize,
            hopSize: hopSize,
            sampleRate: sampleRate
        )
        
        // Initialize K-means clustering
        self.onlineKMeans = OnlineKMeans(k: k, dim: embeddingDim)
        
        // Create Hann window for frame extraction
        let windowSize = frameSize  // Local copy to avoid self capture
        self.windowFunction = (0..<windowSize).map { i in
            let factor = 2.0 * Float.pi * Float(i) / Float(windowSize - 1)
            return 0.5 * (1.0 - cos(factor))
        }
        
        print("🎙️ PhoneticTranscriptionEngine initialized (frameSize: \(frameSize), k: \(k))")
    }
    
    // MARK: - Public Interface
    
    /// Main transcription method
    func transcribe(buffer: AVAudioPCMBuffer, profile: UserProfile? = nil) async -> PhoneticTranscription {
        return await withCheckedContinuation { continuation in
            queue.async {
                Task {
                    let result = await self.processAudioBuffer(buffer)
                    continuation.resume(returning: result)
                }
            }
        }
    }
    
    /// Load symbol mapping for readable output
    func loadSymbolMap(_ mapping: [Int: String]) {
        queue.sync {
            unitToSymbolMap = mapping
        }
        print("📚 Loaded symbol mapping: \(mapping.count) units")
    }
    
    /// Get current codebook for persistence
    func getCodebook() -> CodebookData {
        return onlineKMeans.saveCodebook()
    }
    
    /// Load codebook from persistence
    func loadCodebook(_ data: CodebookData) {
        onlineKMeans.loadCodebook(data)
    }
    
    /// Reset pipeline state
    func reset() {
        vadProcessor.reset()
        onlineKMeans.reset()
        unitToSymbolMap.removeAll()
        print("🎙️ PhoneticTranscriptionEngine reset")
    }
    
    /// Get clustering statistics
    func getStatistics() -> PhoneticEngineStatistics {
        let clusterStats = onlineKMeans.getStatistics()
        return PhoneticEngineStatistics(
            totalFramesProcessed: clusterStats.totalObservations,
            activeClusters: clusterStats.activeClusters,
            clusterUtilization: clusterStats.utilizationRate,
            clusterPurity: clusterStats.clusterPurity,
            hasSymbolMapping: !unitToSymbolMap.isEmpty
        )
    }
    
    // MARK: - Private Processing Methods
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) async -> PhoneticTranscription {
        guard let channelData = buffer.floatChannelData?[0] else {
            return PhoneticTranscription(
                unitIDs: [],
                unitString: "",
                readableSpelling: nil,
                chunkTimes: []
            )
        }
        
        let frameCount = Int(buffer.frameLength)
        
        // Step 1: VAD to find voiced segments
        let voicedSegments = vadProcessor.processBuffer(buffer)
        
        guard !voicedSegments.isEmpty else {
            return PhoneticTranscription(
                unitIDs: [],
                unitString: "",
                readableSpelling: nil,
                chunkTimes: []
            )
        }
        
        // Step 2: Process each voiced segment
        var allUnitIDs: [Int] = []
        var chunkTimes: [Range<TimeInterval>] = []
        
        for segment in voicedSegments {
            guard segment.isValid else { continue }
            
            let segmentResult = await processVoicedSegment(
                channelData: channelData,
                frameCount: frameCount,
                segment: segment
            )
            
            if !segmentResult.unitIDs.isEmpty {
                allUnitIDs.append(contentsOf: segmentResult.unitIDs)
                chunkTimes.append(segment.startTime..<segment.endTime)
            }
        }
        
        // Step 3: Collapse consecutive repeats
        let collapsedUnits = collapseRepeats(allUnitIDs)
        
        // Step 4: Generate string representations
        let unitString = collapsedUnits.map { "U\($0)" }.joined(separator: " ")
        let readableSpelling = generateReadableSpelling(collapsedUnits)
        
        return PhoneticTranscription(
            unitIDs: collapsedUnits,
            unitString: unitString,
            readableSpelling: readableSpelling,
            chunkTimes: chunkTimes
        )
    }
    
    private func processVoicedSegment(
        channelData: UnsafePointer<Float>,
        frameCount: Int,
        segment: VoicedSegment
    ) async -> (unitIDs: [Int], embeddings: [[Float]]) {
        
        let startSample = Int(segment.startTime * sampleRate)
        let endSample = Int(segment.endTime * sampleRate)
        let segmentLength = min(endSample - startSample, frameCount - startSample)
        
        guard segmentLength > frameSize else {
            return (unitIDs: [], embeddings: [])
        }
        
        var unitIDs: [Int] = []
        var embeddings: [[Float]] = []
        
        // Extract overlapping frames
        var frameStart = startSample
        while frameStart + frameSize <= startSample + segmentLength {
            // Extract and window frame
            let frame = Array(UnsafeBufferPointer(start: channelData + frameStart, count: frameSize))
            let windowedFrame = applyWindow(frame)
            
            // Create frame buffer for embedding extraction
            if let frameBuffer = createFrameBuffer(windowedFrame) {
                // Extract embedding using tiling approach
                if let embedding = await extractFrameEmbedding(frameBuffer) {
                    // Quantize to cluster ID
                    let unitID = onlineKMeans.observe(embedding)
                    unitIDs.append(unitID)
                    embeddings.append(embedding)
                }
            }
            
            frameStart += hopSize
        }
        
        return (unitIDs: unitIDs, embeddings: embeddings)
    }
    
    private func extractFrameEmbedding(_ frameBuffer: AVAudioPCMBuffer) async -> [Float]? {
        guard let translationEngine = translationEngine else {
            print("❌ TranslationEngine not available for embedding extraction")
            return nil
        }
        
        // Use the new TranslationEngine method for frame-level extraction
        return await translationEngine.extractFrameEmbedding(from: frameBuffer)
    }
    
    private func applyWindow(_ frame: [Float]) -> [Float] {
        guard frame.count == frameSize else { return frame }
        
        var windowedFrame = [Float](repeating: 0.0, count: frameSize)
        vDSP_vmul(frame, 1, windowFunction, 1, &windowedFrame, 1, vDSP_Length(frameSize))
        return windowedFrame
    }
    
    private func createFrameBuffer(_ frame: [Float]) -> AVAudioPCMBuffer? {
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frame.count)) else {
            return nil
        }
        
        buffer.frameLength = AVAudioFrameCount(frame.count)
        
        guard let channelData = buffer.floatChannelData?[0] else {
            return nil
        }
        
        // Copy frame data
        for i in 0..<frame.count {
            channelData[i] = frame[i]
        }
        
        return buffer
    }
    
    private func collapseRepeats(_ unitIDs: [Int]) -> [Int] {
        guard !unitIDs.isEmpty else { return [] }
        
        var collapsed: [Int] = [unitIDs[0]]
        
        for i in 1..<unitIDs.count {
            if unitIDs[i] != unitIDs[i-1] {
                collapsed.append(unitIDs[i])
            }
        }
        
        return collapsed
    }
    
    private func generateReadableSpelling(_ unitIDs: [Int]) -> String? {
        guard !unitToSymbolMap.isEmpty else { return nil }
        
        let symbols = unitIDs.compactMap { unitToSymbolMap[$0] }
        return symbols.isEmpty ? nil : symbols.joined(separator: " ")
    }
}

// MARK: - Support Structures

struct PhoneticTranscription {
    let unitIDs: [Int]              // Collapsed sequence of unit IDs
    let unitString: String          // "U12 U7 U3" format
    let readableSpelling: String?   // Optional human-readable symbols
    let chunkTimes: [Range<TimeInterval>]  // Timing of voiced segments
    
    var isEmpty: Bool {
        return unitIDs.isEmpty
    }
    
    var duration: TimeInterval {
        return chunkTimes.map { $0.upperBound - $0.lowerBound }.reduce(0, +)
    }
}

struct PhoneticEngineStatistics {
    let totalFramesProcessed: Int
    let activeClusters: Int
    let clusterUtilization: Float
    let clusterPurity: Float
    let hasSymbolMapping: Bool
    
    var isHealthy: Bool {
        return activeClusters > 10 && clusterUtilization > 0.1 && clusterPurity > 0.3
    }
}

// MARK: - UserProfile Extension (placeholder for future use)
// UserProfile and PhoneticProfileData are defined in TrainingExample.swift