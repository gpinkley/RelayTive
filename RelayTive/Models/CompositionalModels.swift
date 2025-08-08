//
//  CompositionalModels.swift  
//  RelayTive
//
//  Core data models for compositional pattern recognition system
//

import Foundation

// MARK: - Audio Segment

/// Represents a temporal segment of audio with its HuBERT embeddings
struct AudioSegment: Identifiable, Codable {
    let id = UUID()
    let startTime: TimeInterval      // Start time in seconds within original audio
    let endTime: TimeInterval        // End time in seconds within original audio  
    let audioData: Data             // Raw audio data for this segment
    let embeddings: [Float]         // HuBERT embeddings for this segment
    let parentExampleId: UUID       // ID of the TrainingExample this segment came from
    let confidence: Float           // Confidence in the segmentation quality (0.0-1.0)
    
    var duration: TimeInterval {
        return endTime - startTime
    }
    
    var isValid: Bool {
        return startTime < endTime && !embeddings.isEmpty && confidence > 0.1
    }
}

// MARK: - Compositional Pattern

/// Represents a discovered pattern that appears across multiple training examples
struct CompositionalPattern: Identifiable, Codable {
    let id = UUID()
    let representativeEmbedding: [Float]    // Average embedding representing this pattern
    let frequency: Int                      // How many times this pattern appears
    let confidence: Float                   // Overall confidence in this pattern (0.0-1.0)
    let averagePosition: Float              // Average position within utterances (0.0-1.0)
    let associatedMeanings: [String]        // Meanings from examples containing this pattern
    let contributingSegments: [UUID]        // IDs of AudioSegments that form this pattern
    let createdAt: Date                     // When this pattern was discovered
    let lastUpdated: Date                   // When this pattern was last reinforced
    
    init(representativeEmbedding: [Float], segments: [AudioSegment], associatedMeanings: [String]) {
        self.representativeEmbedding = representativeEmbedding
        self.frequency = segments.count
        self.confidence = min(1.0, Float(segments.count) * 0.1) // Base confidence on frequency
        self.averagePosition = segments.map { Float($0.startTime) }.reduce(0, +) / Float(segments.count)
        self.associatedMeanings = associatedMeanings
        self.contributingSegments = segments.map { $0.id }
        self.createdAt = Date()
        self.lastUpdated = Date()
    }
    
    var isSignificant: Bool {
        return frequency >= 2 && confidence > 0.3
    }
}

// MARK: - Pattern Collection

/// Collection of all discovered compositional patterns with management utilities
struct PatternCollection: Codable {
    private(set) var patterns: [CompositionalPattern] = []
    private(set) var lastDiscoveryRun: Date?
    
    mutating func addPattern(_ pattern: CompositionalPattern) {
        // Avoid duplicates based on embedding similarity
        let isDuplicate = patterns.contains { existingPattern in
            cosineSimilarity(pattern.representativeEmbedding, existingPattern.representativeEmbedding) > 0.95
        }
        
        if !isDuplicate {
            patterns.append(pattern)
            patterns.sort { $0.confidence > $1.confidence } // Keep highest confidence first
        }
    }
    
    mutating func updatePattern(id: UUID, newSegments: [AudioSegment], newMeanings: [String]) {
        if let index = patterns.firstIndex(where: { $0.id == id }) {
            let existingPattern = patterns[index]
            let updatedPattern = CompositionalPattern(
                representativeEmbedding: existingPattern.representativeEmbedding,
                segments: newSegments,
                associatedMeanings: Array(Set(existingPattern.associatedMeanings + newMeanings))
            )
            patterns[index] = updatedPattern
        }
    }
    
    mutating func removeWeakPatterns() {
        patterns.removeAll { !$0.isSignificant }
    }
    
    mutating func markDiscoveryComplete() {
        lastDiscoveryRun = Date()
    }
    
    var significantPatterns: [CompositionalPattern] {
        return patterns.filter { $0.isSignificant }
    }
    
    func findSimilarPatterns(to embedding: [Float], threshold: Float = 0.7) -> [(pattern: CompositionalPattern, similarity: Float)] {
        return patterns.compactMap { pattern in
            let similarity = cosineSimilarity(embedding, pattern.representativeEmbedding)
            return similarity > threshold ? (pattern, similarity) : nil
        }.sorted { $0.similarity > $1.similarity }
    }
}

// MARK: - Pattern Match Result

/// Result of matching audio against compositional patterns
struct PatternMatchResult {
    let matchedPatterns: [(pattern: CompositionalPattern, confidence: Float, position: TimeInterval)]
    let overallConfidence: Float
    let reconstructedTranslation: String
    let explanation: String
    
    var hasMatches: Bool {
        return !matchedPatterns.isEmpty && overallConfidence > 0.5
    }
}

// MARK: - Segmentation Strategy

enum SegmentationStrategy {
    case fixed(duration: TimeInterval)           // Fixed-length segments
    case variable(minDuration: TimeInterval, maxDuration: TimeInterval)  // Variable-length segments
    case adaptive                               // Adaptive based on audio features
    
    var defaultDuration: TimeInterval {
        switch self {
        case .fixed(let duration):
            return duration
        case .variable(let min, _):
            return min
        case .adaptive:
            return 0.5 // Default for adaptive
        }
    }
}

// MARK: - Pattern Discovery Configuration  

struct PatternDiscoveryConfig {
    let minPatternFrequency: Int            // Minimum times pattern must appear
    let minPatternConfidence: Float         // Minimum confidence threshold
    let similarityThreshold: Float          // Embedding similarity threshold
    let maxPatternsToDiscover: Int         // Limit on total patterns
    let segmentationStrategy: SegmentationStrategy
    
    static let `default` = PatternDiscoveryConfig(
        minPatternFrequency: 2,
        minPatternConfidence: 0.3,
        similarityThreshold: 0.75,
        maxPatternsToDiscover: 100,
        segmentationStrategy: .variable(minDuration: 0.2, maxDuration: 1.0)
    )
}

// MARK: - Utility Functions

/// Calculate cosine similarity between two embedding vectors
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count && !a.isEmpty else { return 0.0 }
    
    let dotProduct = zip(a, b).map(*).reduce(0, +)
    let magnitudeA = sqrt(a.map { $0 * $0 }.reduce(0, +))
    let magnitudeB = sqrt(b.map { $0 * $0 }.reduce(0, +))
    
    guard magnitudeA > 0 && magnitudeB > 0 else { return 0.0 }
    
    return dotProduct / (magnitudeA * magnitudeB)
}

/// Calculate average embedding from a collection of embeddings
func averageEmbedding(from embeddings: [[Float]]) -> [Float]? {
    guard !embeddings.isEmpty,
          let firstEmbedding = embeddings.first,
          embeddings.allSatisfy({ $0.count == firstEmbedding.count }) else {
        return nil
    }
    
    let dimensionCount = firstEmbedding.count
    var averages = Array<Float>(repeating: 0.0, count: dimensionCount)
    
    for embedding in embeddings {
        for (index, value) in embedding.enumerated() {
            averages[index] += value
        }
    }
    
    let count = Float(embeddings.count)
    return averages.map { $0 / count }
}