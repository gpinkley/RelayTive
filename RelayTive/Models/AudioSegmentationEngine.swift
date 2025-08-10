//
//  AudioSegmentationEngine.swift
//  RelayTive
//
//  Engine for extracting temporal audio segments and computing their HuBERT embeddings
//

import Foundation
import AVFoundation
import Accelerate

/// Diagnostic information for segment processing
struct SegmentDiag: Codable {
    let idx: Int
    let start: TimeInterval
    let end: TimeInterval
    let frames: Int
    let rms: Float
    let pad: Int
    let success: Bool
    let reason: String
}

/// Engine responsible for segmenting audio and extracting embeddings for each segment
class AudioSegmentationEngine {
    
    // MARK: - Tuning Constants
    struct SegmentationTuning {
        static let minDuration: TimeInterval = 0.5
        static let maxDuration: TimeInterval = 2.0
        static let frameDuration: TimeInterval = 0.050
        static let stepDuration: TimeInterval = 0.025
        static let similarityThreshold: Float = 0.75
    }
    private let translationEngine: TranslationEngine
    private let audioFormat = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
    
    init(translationEngine: TranslationEngine) {
        self.translationEngine = translationEngine
    }
    
    // MARK: - Main Segmentation Method
    
    /// Extract temporal segments from a training example and compute their embeddings
    func extractSegments(from trainingExample: TrainingExample, 
                        strategy: SegmentationStrategy = .embeddingBased(
                            minDuration: SegmentationTuning.minDuration,
                            maxDuration: SegmentationTuning.maxDuration,
                            similarityThreshold: SegmentationTuning.similarityThreshold
                        )) async -> [AudioSegment] {
        
        print("🔍 Extracting segments from training example: \(trainingExample.typicalExplanation)")
        
        // Prefer file URL over embedded data for full recording decode
        var mutableExample = trainingExample
        let audioBuffer: AVAudioPCMBuffer?
        
        if let fileURL = mutableExample.getOrCreateAudioFileURL() {
            audioBuffer = decodeAudioFile(from: fileURL)
        } else if let audioData = trainingExample.audioData {
            audioBuffer = createAudioBuffer(from: audioData)
        } else {
            print("❌ No audio data available for training example")
            return []
        }
        
        guard let buffer = audioBuffer else {
            print("❌ Failed to create audio buffer from training example")
            return []
        }
        
        // Determine segmentation points based on strategy
        let segmentRanges = await determineSegmentRanges(audioBuffer: buffer, strategy: strategy)
        print("📊 Found \(segmentRanges.count) potential segments")
        
        // Extract segments and compute embeddings - ALWAYS produce embeddings via zero-padding
        var segments: [AudioSegment] = []
        var diagnostics: [SegmentDiag] = []
        
        for (index, range) in segmentRanges.enumerated() {
            if Log.isVerbose {
                print("⏱️ Processing segment \(index + 1)/\(segmentRanges.count): \(range.startTime)s-\(range.endTime)s")
            }
            
            var segmentDiag = SegmentDiag(
                idx: index + 1,
                start: range.startTime,
                end: range.endTime,
                frames: 0,
                rms: 0,
                pad: 0,
                success: false,
                reason: "unknown"
            )

            if let segmentAudio = extractAudioSegment(from: buffer, range: range) {
                // Calculate buffer diagnostics
                let frameCount = Int(segmentAudio.frameLength)
                let bufferAnalysis = analyzeAudioBuffer(segmentAudio)
                
                // ALWAYS extract embeddings - unified preprocessing handles zero-padding
                if let embeddings = await translationEngine.extractEmbeddings(from: segmentAudio) {
                    // Successful segment creation
                    autoreleasepool {
                        let confidence = calculateSegmentConfidence(audioBuffer: segmentAudio, range: range)

                        #if DEBUG
                        let segmentData = audioBufferToData(segmentAudio)
                        #else
                        let segmentData: Data? = nil
                        #endif

                        let segment = AudioSegment(
                            startTime: range.startTime,
                            endTime: range.endTime,
                            audioData: segmentData,
                            embeddings: embeddings,
                            parentExampleId: trainingExample.id,
                            confidence: confidence
                        )
                        segments.append(segment)
                    }
                    
                    segmentDiag = SegmentDiag(
                        idx: index + 1,
                        start: range.startTime,
                        end: range.endTime,
                        frames: frameCount,
                        rms: bufferAnalysis.rms,
                        pad: bufferAnalysis.padCount,
                        success: true,
                        reason: "ok"
                    )
                    
                    print("✅ Created segment with \(embeddings.count) embedding dimensions")
                } else {
                    segmentDiag = SegmentDiag(
                        idx: index + 1,
                        start: range.startTime,
                        end: range.endTime,
                        frames: frameCount,
                        rms: bufferAnalysis.rms,
                        pad: bufferAnalysis.padCount,
                        success: false,
                        reason: "embedding_failed"
                    )
                    print("⚠️ Segment \(index + 1) embedding extraction failed but continuing")
                }
            } else {
                segmentDiag = SegmentDiag(
                    idx: index + 1,
                    start: range.startTime,
                    end: range.endTime,
                    frames: 0,
                    rms: 0,
                    pad: 0,
                    success: false,
                    reason: "buffer_extraction_failed"
                )
                print("⚠️ Segment \(index + 1) buffer extraction failed but continuing")
            }
            
            diagnostics.append(segmentDiag)
        }
        
        // Write diagnostics JSON report
        writeDiagnosticsReport(diagnostics: diagnostics, trainingExample: trainingExample)
       
        // Print segmentation summary
        printSegmentationSummary(segments.filter { $0.isValid })
        
        return segments.filter { $0.isValid }
    }
    
    // MARK: - Public Embedding Extraction
    
    /// Extract embeddings from raw audio data (exposed for CompositionalMatcher fallback)
    func extractEmbeddings(from audioData: Data) async -> [Float]? {
        return await translationEngine.extractEmbeddings(audioData)
    }
    
    /// Extract embeddings from audio buffer (preferred method)
    func extractEmbeddings(from buffer: AVAudioPCMBuffer) async -> [Float]? {
        return await translationEngine.extractEmbeddings(from: buffer)
    }
    
    // MARK: - Audio Processing
    
    /// Decode audio file to PCM buffer with real length (no padding)
    private func decodeAudioFile(from url: URL) -> AVAudioPCMBuffer? {
        do {
            let audioFile = try AVAudioFile(forReading: url)
            let frameCount = AVAudioFrameCount(audioFile.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFile.processingFormat, frameCapacity: frameCount) else {
                print("❌ Failed to create PCM buffer for audio file")
                return nil
            }
            
            try audioFile.read(into: buffer)
            print("✅ Decoded audio file: \(frameCount) frames, \(Double(frameCount) / audioFile.processingFormat.sampleRate)s duration")
            return buffer
        } catch {
            print("❌ Failed to decode audio file: \(error)")
            return nil
        }
    }
    
    // Deprecated - prefer extractEmbeddings(from: AVAudioPCMBuffer) for internal segments
    private func createAudioBuffer(from audioData: Data) -> AVAudioPCMBuffer? {
        print("⚠️ Using deprecated createAudioBuffer in AudioSegmentationEngine")
        
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
    
    private func audioBufferToData(_ buffer: AVAudioPCMBuffer) -> Data? {
        guard let floatData = buffer.floatChannelData?[0] else { return nil }
        
        let frameCount = Int(buffer.frameLength)
        var int16Data = Data()
        
        for i in 0..<frameCount {
            let sample = Int16(max(-32768, min(32767, floatData[i] * 32768.0)))
            int16Data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }
        
        return int16Data
    }
    
    // MARK: - Segmentation Strategies
    
    private func determineSegmentRanges(audioBuffer: AVAudioPCMBuffer, strategy: SegmentationStrategy) async -> [SegmentRange] {
        let totalDuration = Double(audioBuffer.frameLength) / audioBuffer.format.sampleRate
        
        switch strategy {
        case .fixed(let window, _):
            return createFixedSegments(totalDuration: totalDuration, segmentDuration: window)
            
        case .variable(let minDuration, let maxDuration, _):
            return createVariableSegments(audioBuffer: audioBuffer, 
                                        totalDuration: totalDuration, 
                                        minDuration: minDuration, 
                                        maxDuration: maxDuration)
            
        case .adaptive:
            return createAdaptiveSegments(audioBuffer: audioBuffer, totalDuration: totalDuration)
            
        case .embeddingBased(let minDuration, let maxDuration, let similarityThreshold):
            #if DEBUG
            print("⚠️ Using energy-adaptive segmentation in DEBUG mode instead of embedding-based")
            return createAdaptiveSegments(audioBuffer: audioBuffer, totalDuration: totalDuration)
            #else
            return await createEmbeddingBasedSegments(audioBuffer: audioBuffer, 
                                                    totalDuration: totalDuration, 
                                                    minDuration: minDuration, 
                                                    maxDuration: maxDuration, 
                                                    similarityThreshold: similarityThreshold)
            #endif
        }
    }
    
    private func createFixedSegments(totalDuration: TimeInterval, segmentDuration: TimeInterval) -> [SegmentRange] {
        var ranges: [SegmentRange] = []
        var currentTime: TimeInterval = 0
        
        while currentTime < totalDuration {
            let endTime = min(currentTime + segmentDuration, totalDuration)
            ranges.append(SegmentRange(startTime: currentTime, endTime: endTime))
            currentTime = endTime
        }
        
        return ranges
    }
    
    private func createVariableSegments(audioBuffer: AVAudioPCMBuffer, 
                                      totalDuration: TimeInterval, 
                                      minDuration: TimeInterval, 
                                      maxDuration: TimeInterval) -> [SegmentRange] {
        // Create reasonable number of non-overlapping segments
        var ranges: [SegmentRange] = []
        
        // Limit total segments to prevent infinite loops
        let maxSegments = min(10, Int(totalDuration / minDuration))
        let segmentDuration = totalDuration / Double(maxSegments)
        
        for i in 0..<maxSegments {
            let startTime = Double(i) * segmentDuration
            let endTime = min(startTime + segmentDuration, totalDuration)
            
            if endTime - startTime >= minDuration {
                ranges.append(SegmentRange(startTime: startTime, endTime: endTime))
            }
        }
        
        return ranges
    }
    
    private func createAdaptiveSegments(audioBuffer: AVAudioPCMBuffer, totalDuration: TimeInterval) -> [SegmentRange] {
        // Use energy-based segmentation for adaptive strategy
        let energyProfile = calculateEnergyProfile(audioBuffer: audioBuffer)
        let segmentBoundaries = findEnergyBasedBoundaries(energyProfile: energyProfile, totalDuration: totalDuration)
        
        var ranges: [SegmentRange] = []
        for i in 0..<segmentBoundaries.count - 1 {
            ranges.append(SegmentRange(startTime: segmentBoundaries[i], endTime: segmentBoundaries[i + 1]))
        }
        
        return ranges
    }
    
    private func calculateEnergyProfile(audioBuffer: AVAudioPCMBuffer) -> [Float] {
        guard let floatData = audioBuffer.floatChannelData?[0] else { return [] }
        
        let frameCount = Int(audioBuffer.frameLength)
        let windowSize = 1024  // ~64ms at 16kHz
        let stepSize = windowSize / 2  // 50% overlap
        
        var energyProfile: [Float] = []
        
        for start in stride(from: 0, to: frameCount - windowSize, by: stepSize) {
            var energy: Float = 0
            for i in start..<min(start + windowSize, frameCount) {
                let sample = floatData[i]
                energy += sample * sample
            }
            energyProfile.append(energy / Float(windowSize))
        }
        
        return energyProfile
    }
    
    private func findEnergyBasedBoundaries(energyProfile: [Float], totalDuration: TimeInterval) -> [TimeInterval] {
        guard !energyProfile.isEmpty else { return [0, totalDuration] }
        
        // Find local minima in energy as potential segment boundaries
        var boundaries: [TimeInterval] = [0] // Always start at 0
        
        let windowSize = max(5, energyProfile.count / 20) // Adaptive window size
        
        for i in windowSize..<(energyProfile.count - windowSize) {
            let current = energyProfile[i]
            let isLocalMinimum = (i-windowSize..<i).allSatisfy { energyProfile[$0] >= current } &&
                               (i+1..<i+windowSize+1).allSatisfy { energyProfile[$0] >= current }
            
            if isLocalMinimum && current < energyProfile.reduce(0, +) / Float(energyProfile.count) * 0.3 {
                let timePoint = (Double(i) / Double(energyProfile.count)) * totalDuration
                boundaries.append(timePoint)
            }
        }
        
        boundaries.append(totalDuration) // Always end at total duration
        return boundaries.sorted()
    }
    
    // MARK: - Segment Extraction
    
    private func extractAudioSegment(from buffer: AVAudioPCMBuffer, range: SegmentRange) -> AVAudioPCMBuffer? {
        let sampleRate = buffer.format.sampleRate
        let startFrame = Int(range.startTime * sampleRate)
        let endFrame = Int(range.endTime * sampleRate)
        let frameCount = endFrame - startFrame
        
        guard startFrame >= 0, 
              endFrame <= buffer.frameLength, 
              frameCount > 0,
              let segmentBuffer = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return nil
        }
        
        segmentBuffer.frameLength = AVAudioFrameCount(frameCount)
        
        guard let sourceData = buffer.floatChannelData?[0],
              let destData = segmentBuffer.floatChannelData?[0] else {
            return nil
        }
        
        // Copy the segment data
        memcpy(destData, sourceData.advanced(by: startFrame), frameCount * MemoryLayout<Float>.size)
        
        return segmentBuffer
    }
    
    private func calculateSegmentConfidence(audioBuffer: AVAudioPCMBuffer, range: SegmentRange) -> Float {
        guard let floatData = audioBuffer.floatChannelData?[0] else { return 0.0 }
        
        let frameCount = Int(audioBuffer.frameLength)
        
        // Calculate RMS energy
        var rmsEnergy: Float = 0
        for i in 0..<frameCount {
            let sample = floatData[i]
            rmsEnergy += sample * sample
        }
        rmsEnergy = sqrt(rmsEnergy / Float(frameCount))
        
        // Calculate segment duration factor
        let duration = range.duration
        let durationFactor: Float = duration > 0.1 && duration < 2.0 ? 1.0 : 0.5
        
        // Combine factors for overall confidence
        let energyFactor = min(1.0, rmsEnergy * 10) // Scale to 0-1 range
        
        return energyFactor * durationFactor
    }
    
    // MARK: - Embedding Extraction
    
    // Deprecated - use buffer-based extraction instead
    private func extractEmbeddingsForSegment(_ audioData: Data) async -> [Float]? {
        return await translationEngine.extractEmbeddings(audioData)
    }
    
    // MARK: - Embedding-Based Segmentation
    
    private func createEmbeddingBasedSegments(audioBuffer: AVAudioPCMBuffer,
                                            totalDuration: TimeInterval,
                                            minDuration: TimeInterval,
                                            maxDuration: TimeInterval,
                                            similarityThreshold: Float) async -> [SegmentRange] {
        
        let frameEmbeddings = await extractFrameLevelEmbeddings(audioBuffer: audioBuffer)
        
        guard frameEmbeddings.count >= 4 else {
            print("⚠️ Too few frame embeddings (\(frameEmbeddings.count)), falling back to fixed segments")
            return createFixedSegments(totalDuration: totalDuration, segmentDuration: 0.5)
        }
        
        // Cap frame embeddings to prevent excessive processing
        let cappedEmbeddings = frameEmbeddings.count > 40 ? Array(frameEmbeddings.prefix(40)) : frameEmbeddings
        if frameEmbeddings.count > 40 {
            print("⚠️ Capped embeddings at 40 frames to avoid excessive processing")
        }
        
        let boundaries = findEmbeddingSimilarityBoundaries(embeddings: cappedEmbeddings, 
                                                         totalDuration: totalDuration,
                                                         similarityThreshold: similarityThreshold)
        
        let segments = createSegmentsFromBoundaries(boundaries: boundaries,
                                                  totalDuration: totalDuration,
                                                  minDuration: minDuration,
                                                  maxDuration: maxDuration)
        
        return segments
    }
    
    private func extractFrameLevelEmbeddings(audioBuffer: AVAudioPCMBuffer) async -> [[Float]] {
        let sampleRate = audioBuffer.format.sampleRate
        let frameDuration = SegmentationTuning.frameDuration
        let stepDuration = SegmentationTuning.stepDuration
        let totalDuration = Double(audioBuffer.frameLength) / sampleRate
        
        var embeddings: [[Float]] = []
        var currentTime: TimeInterval = 0
        var totalFrames = 0
        
        while currentTime + frameDuration <= totalDuration {
            let endTime = min(currentTime + frameDuration, totalDuration)
            totalFrames += 1
            
            if let frameBuffer = extractAudioSegment(from: audioBuffer, 
                                                   range: SegmentRange(startTime: currentTime, endTime: endTime)) {
                // ALWAYS extract embeddings - unified preprocessing handles zero-padding
                if let embedding = await translationEngine.extractEmbeddings(from: frameBuffer) {
                    embeddings.append(embedding)
                } else {
                    print("❌ Frame \(totalFrames) embedding extraction failed but continuing")
                }
            } else {
                print("❌ Frame \(totalFrames) segment extraction failed but continuing")
            }
            
            currentTime += stepDuration
        }
        
        print("Extracted embeddings for \(embeddings.count)/\(totalFrames) frames")
        return embeddings
    }
    
    private func findEmbeddingSimilarityBoundaries(embeddings: [[Float]], 
                                                 totalDuration: TimeInterval,
                                                 similarityThreshold: Float) -> [TimeInterval] {
        var boundaries: [TimeInterval] = [0]
        
        let stepDuration = SegmentationTuning.stepDuration
        
        for i in 1..<embeddings.count {
            let similarity = cosineSimilarity(embeddings[i-1], embeddings[i])
            
            if similarity < similarityThreshold {
                let timePoint = Double(i) * stepDuration
                boundaries.append(timePoint)
            }
        }
        
        boundaries.append(totalDuration)
        return boundaries.sorted()
    }
    
    private func createSegmentsFromBoundaries(boundaries: [TimeInterval],
                                            totalDuration: TimeInterval,
                                            minDuration: TimeInterval,
                                            maxDuration: TimeInterval) -> [SegmentRange] {
        var segments: [SegmentRange] = []
        var shortSegmentCount = 0
        
        for i in 0..<boundaries.count - 1 {
            let startTime = boundaries[i]
            var endTime = boundaries[i + 1]
            var duration = endTime - startTime
            
            if duration < minDuration && i < boundaries.count - 2 {
                endTime = boundaries[i + 2]
                duration = endTime - startTime
            }
            
            if duration > maxDuration {
                let numSplits = Int(ceil(duration / maxDuration))
                let splitDuration = duration / Double(numSplits)
                
                for j in 0..<numSplits {
                    let splitStart = startTime + Double(j) * splitDuration
                    let splitEnd = min(startTime + Double(j + 1) * splitDuration, endTime)
                    segments.append(SegmentRange(startTime: splitStart, endTime: splitEnd))
                }
            } else if duration >= minDuration {
                segments.append(SegmentRange(startTime: startTime, endTime: endTime))
            } else {
                shortSegmentCount += 1
            }
        }
        
        let avgDuration = segments.map(\.duration).reduce(0, +) / Double(segments.count)
        
        #if DEBUG
        if shortSegmentCount > 0 || avgDuration < 0.2 {
            print("⚠️ Segmentation quality: \(shortSegmentCount) short segments rejected, avg duration: \(String(format: "%.3f", avgDuration))s")
            if avgDuration < 0.2 {
                print("💡 Consider raising similarityThreshold or minDuration")
            }
        }
        #endif
        
        return segments
    }
    
    private func printSegmentationSummary(_ segments: [AudioSegment]) {
        guard !segments.isEmpty else { 
            print("Segmentation: n=0 total=0s")
            return 
        }
        
        let durations = segments.map { $0.duration }
        let count = durations.count
        let total = durations.reduce(0, +)
        let avgDuration = total / Double(count)
        let minDuration = durations.min() ?? 0
        let maxDuration = durations.max() ?? 0
        
        let variance = durations.map { pow($0 - avgDuration, 2) }.reduce(0, +) / Double(count)
        let stdDev = sqrt(variance)
        
        print("Segmentation: n=\(count) avg=\(String(format: "%.3f", avgDuration))s min=\(String(format: "%.3f", minDuration))s max=\(String(format: "%.3f", maxDuration))s std=\(String(format: "%.3f", stdDev))s total=\(String(format: "%.3f", total))s")
    }
    
    /// Analyze audio buffer for diagnostics (no padding detection - that's in TranslationEngine)
    private func analyzeAudioBuffer(_ buffer: AVAudioPCMBuffer) -> (rms: Float, padCount: Int) {
        guard let channelData = buffer.floatChannelData?[0] else {
            return (0, 0)
        }
        
        let frameCount = Int(buffer.frameLength)
        var sumSquares: Float = 0
        
        for i in 0..<frameCount {
            let sample = channelData[i]
            sumSquares += sample * sample
        }
        
        let rms = frameCount > 0 ? sqrt(sumSquares / Float(frameCount)) : 0
        
        // Remove padding guesswork - actual padding is computed in TranslationEngine
        return (rms, 0)
    }
    
    private struct SegReport: Codable {
        let exampleId: String
        let explanation: String
        let timestamp: String
        let segments: [SegmentDiag]
    }
    
    /// Write segment diagnostics to JSON file for debugging
    private func writeDiagnosticsReport(diagnostics: [SegmentDiag], trainingExample: TrainingExample) {
        #if DEBUG
        do {
            let report = SegReport(
                exampleId: trainingExample.id.uuidString,
                explanation: trainingExample.typicalExplanation,
                timestamp: ISO8601DateFormatter().string(from: Date()),
                segments: diagnostics
            )
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try enc.encode(report)
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            try data.write(to: docs.appendingPathComponent("segmentation_diag_\(trainingExample.id.uuidString.prefix(8)).json"))
            print("📄 Diagnostics report written")
        } catch {
            print("⚠️ Failed to write diagnostics report: \(error)")
        }
        #endif
    }
}

// MARK: - Supporting Types

private struct SegmentRange {
    let startTime: TimeInterval
    let endTime: TimeInterval
    
    var duration: TimeInterval {
        return endTime - startTime
    }
}
