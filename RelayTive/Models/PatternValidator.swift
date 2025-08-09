//
//  PatternValidator.swift
//  RelayTive
//
//  Pattern validation logic for compositional pattern discovery
//

import Foundation

struct PatternValidator {
    
    /// Calculate meaning consistency for a pattern based on associated meanings
    /// Returns proportion of the modal meaning among all associated meanings (0..1)
    static func meaningConsistency(for pattern: CompositionalPattern,
                                 examplesById: [UUID: TrainingExample]) -> Float {
        guard !pattern.associatedMeanings.isEmpty else { return 0.0 }
        
        // Count frequency of each meaning
        var meaningCounts: [String: Int] = [:]
        for meaning in pattern.associatedMeanings {
            meaningCounts[meaning, default: 0] += 1
        }
        
        // Find the most frequent meaning
        let maxCount = meaningCounts.values.max() ?? 0
        let totalMeanings = pattern.associatedMeanings.count
        
        return Float(maxCount) / Float(totalMeanings)
    }
    
    /// Calculate embedding cohesion for a pattern
    /// Returns average cosine similarity between contributing segments and representative embedding
    static func cohesion(for pattern: CompositionalPattern,
                        segmentsById: [UUID: AudioSegment]) -> Float {
        guard !pattern.contributingSegments.isEmpty else { return 0.0 }
        
        var similarities: [Float] = []
        
        for segmentId in pattern.contributingSegments {
            guard let segment = segmentsById[segmentId] else { continue }
            
            let similarity = cosineSimilarity(segment.embeddings, pattern.representativeEmbedding)
            similarities.append(similarity)
        }
        
        guard !similarities.isEmpty else { return 0.0 }
        
        return similarities.reduce(0, +) / Float(similarities.count)
    }
    
    /// Determine if a pattern is valid based on comprehensive criteria
    static func isValid(_ pattern: CompositionalPattern,
                       segmentsById: [UUID: AudioSegment],
                       examplesById: [UUID: TrainingExample],
                       cfg: PatternDiscoveryConfig) -> Bool {
        
        // Check basic frequency and confidence requirements
        guard pattern.frequency >= cfg.minPatternFrequency else { return false }
        guard pattern.confidence >= cfg.minPatternConfidence else { return false }
        
        // Check embedding cohesion
        let cohesionScore = cohesion(for: pattern, segmentsById: segmentsById)
        guard cohesionScore >= cfg.similarityThreshold else { return false }
        
        // Check meaning consistency
        let consistencyScore = meaningConsistency(for: pattern, examplesById: examplesById)
        guard consistencyScore >= cfg.meaningConsistencyThreshold else { return false }
        
        return true
    }
}