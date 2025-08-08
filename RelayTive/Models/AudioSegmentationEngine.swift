//
//  AudioSegmentationEngine.swift
//  RelayTive
//
//  Engine for extracting temporal audio segments and computing their HuBERT embeddings
//

import Foundation
import AVFoundation
import Accelerate

/// Engine responsible for segmenting audio and extracting embeddings for each segment
class AudioSegmentationEngine {
    private let translationEngine: TranslationEngine
    private let audioFormat = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
    
    init(translationEngine: TranslationEngine) {
        self.translationEngine = translationEngine
    }
    
    // MARK: - Main Segmentation Method
    
    /// Extract temporal segments from a training example and compute their embeddings
    func extractSegments(from trainingExample: TrainingExample, 
                        strategy: SegmentationStrategy = .variable(minDuration: 0.2, maxDuration: 1.0)) async -> [AudioSegment] {
        
        print("🔍 Extracting segments from training example: \(trainingExample.typicalExplanation)")
        
        // Convert audio data to processable format
        guard let audioBuffer = createAudioBuffer(from: trainingExample.atypicalAudio) else {
            print("❌ Failed to create audio buffer from training example")
            return []
        }
        
        // Determine segmentation points based on strategy
        let segmentRanges = determineSegmentRanges(audioBuffer: audioBuffer, strategy: strategy)
        print("📊 Found \(segmentRanges.count) potential segments")
        
        // Extract segments and compute embeddings
        var segments: [AudioSegment] = []
        
        for (index, range) in segmentRanges.enumerated() {
            print("⏱️ Processing segment \(index + 1)/\(segmentRanges.count): \(range.startTime)s-\(range.endTime)s")
            
            if let segmentAudio = extractAudioSegment(from: audioBuffer, range: range),
               let segmentData = audioBufferToData(segmentAudio),
               let embeddings = await extractEmbeddingsForSegment(segmentData) {
                
                let confidence = calculateSegmentConfidence(audioBuffer: segmentAudio, range: range)
                
                let segment = AudioSegment(
                    startTime: range.startTime,
                    endTime: range.endTime,
                    audioData: segmentData,
                    embeddings: embeddings,
                    parentExampleId: trainingExample.id,
                    confidence: confidence
                )
                
                segments.append(segment)
                print("✅ Created segment with \(embeddings.count) embedding dimensions, confidence: \(confidence)")
            } else {
                print("❌ Failed to process segment \(index + 1)")
            }
        }
        
        print("🎯 Successfully extracted \(segments.count) valid segments")
        return segments.filter { $0.isValid }
    }
    
    // MARK: - Audio Processing
    
    private func createAudioBuffer(from audioData: Data) -> AVAudioPCMBuffer? {
        // Convert raw audio data to AVAudioPCMBuffer
        // Assume input is 16kHz, 16-bit mono PCM
        let frameCount = audioData.count / 2 // 2 bytes per 16-bit sample
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(frameCount)) else {
            print("Failed to create PCM buffer")
            return nil
        }
        
        buffer.frameLength = AVAudioFrameCount(frameCount)
        
        // Copy audio data to buffer
        audioData.withUnsafeBytes { bytes in
            let int16Ptr = bytes.bindMemory(to: Int16.self)
            let floatPtr = buffer.floatChannelData![0]
            
            // Convert Int16 to Float and normalize to [-1.0, 1.0]
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
    
    private func determineSegmentRanges(audioBuffer: AVAudioPCMBuffer, strategy: SegmentationStrategy) -> [SegmentRange] {
        let totalDuration = Double(audioBuffer.frameLength) / audioBuffer.format.sampleRate
        
        switch strategy {
        case .fixed(let duration):
            return createFixedSegments(totalDuration: totalDuration, segmentDuration: duration)
            
        case .variable(let minDuration, let maxDuration):
            return createVariableSegments(audioBuffer: audioBuffer, 
                                        totalDuration: totalDuration, 
                                        minDuration: minDuration, 
                                        maxDuration: maxDuration)
            
        case .adaptive:
            return createAdaptiveSegments(audioBuffer: audioBuffer, totalDuration: totalDuration)
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
        // Create overlapping variable-length segments
        var ranges: [SegmentRange] = []
        let stepSize = minDuration * 0.5 // 50% overlap
        
        var currentTime: TimeInterval = 0
        
        while currentTime < totalDuration {
            // Vary segment duration based on position and some randomness
            let progress = currentTime / totalDuration
            let targetDuration = minDuration + (maxDuration - minDuration) * (0.5 + 0.5 * sin(progress * .pi))
            let endTime = min(currentTime + targetDuration, totalDuration)
            
            if endTime - currentTime >= minDuration * 0.5 { // Ensure minimum viable duration
                ranges.append(SegmentRange(startTime: currentTime, endTime: endTime))
            }
            
            currentTime += stepSize
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
    
    private func extractEmbeddingsForSegment(_ audioData: Data) async -> [Float]? {
        return await translationEngine.extractEmbeddings(audioData)
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