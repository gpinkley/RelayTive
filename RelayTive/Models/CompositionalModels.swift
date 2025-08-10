//
//  CompositionalModels.swift
//  RelayTive
//

import Foundation

// ============================================================
// MARK: - AudioSegment
// ============================================================

struct AudioSegment: Identifiable, Codable {
    let id: UUID
    let startTime: TimeInterval
    let endTime: TimeInterval
    let audioData: Data?
    let embeddings: [Float]
    let parentExampleId: UUID
    let confidence: Float

    var duration: TimeInterval { endTime - startTime }
    var isValid: Bool { startTime < endTime && !embeddings.isEmpty && confidence > 0.1 }

    // Creation init for new segments
    init(startTime: TimeInterval,
         endTime: TimeInterval,
         audioData: Data? = nil,
         embeddings: [Float],
         parentExampleId: UUID,
         confidence: Float = 1.0) {
        self.id = UUID()
        self.startTime = startTime
        self.endTime = endTime
        self.audioData = audioData
        self.embeddings = embeddings
        self.parentExampleId = parentExampleId
        self.confidence = confidence
    }

    // Uniform immutable copier
    func with(
        startTime: TimeInterval? = nil,
        endTime: TimeInterval? = nil,
        audioData: Data? = nil,
        embeddings: [Float]? = nil,
        parentExampleId: UUID? = nil,
        confidence: Float? = nil
    ) -> AudioSegment {
        return AudioSegment(
            id: id,
            startTime: startTime ?? self.startTime,
            endTime: endTime ?? self.endTime,
            audioData: audioData ?? self.audioData,
            embeddings: embeddings ?? self.embeddings,
            parentExampleId: parentExampleId ?? self.parentExampleId,
            confidence: confidence ?? self.confidence
        )
    }

    // Private memberwise used by with(...)
    private init(id: UUID,
                 startTime: TimeInterval,
                 endTime: TimeInterval,
                 audioData: Data?,
                 embeddings: [Float],
                 parentExampleId: UUID,
                 confidence: Float) {
        self.id = id
        self.startTime = startTime
        self.endTime = endTime
        self.audioData = audioData
        self.embeddings = embeddings
        self.parentExampleId = parentExampleId
        self.confidence = confidence
    }
}

// ============================================================
// MARK: - CompositionalPattern
// ============================================================

struct CompositionalPattern: Identifiable, Codable {
    let id: UUID
    let representativeEmbedding: [Float]
    let frequency: Int
    let confidence: Float
    let averagePosition: Float      // 0 start .. 1 end
    let associatedMeanings: [String]
    let contributingSegments: [UUID]
    let createdAt: Date
    let lastUpdated: Date

    var isSignificant: Bool { confidence >= 0.5 && frequency >= 2 }

    // Convenience init used by discovery
    init(
        representativeEmbedding: [Float],
        segments: [AudioSegment],
        associatedMeanings: [String],
        confidence: Float,
        createdAt: Date = Date()
    ) {
        self.id = UUID()
        self.representativeEmbedding = representativeEmbedding
        self.frequency = segments.count
        self.confidence = confidence
        let posSum = segments.map { Float($0.startTime) }.reduce(0, +)
        self.averagePosition = segments.isEmpty ? 0 : posSum / Float(segments.count)
        self.associatedMeanings = associatedMeanings
        self.contributingSegments = segments.map(\.id)
        self.createdAt = createdAt
        self.lastUpdated = createdAt
    }

    // Uniform immutable copier
    func with(
        representativeEmbedding: [Float]? = nil,
        frequency: Int? = nil,
        confidence: Float? = nil,
        averagePosition: Float? = nil,
        associatedMeanings: [String]? = nil,
        contributingSegments: [UUID]? = nil,
        lastUpdated: Date = Date()
    ) -> CompositionalPattern {
        CompositionalPattern(
            id: id,
            representativeEmbedding: representativeEmbedding ?? self.representativeEmbedding,
            frequency: frequency ?? self.frequency,
            confidence: confidence ?? self.confidence,
            averagePosition: averagePosition ?? self.averagePosition,
            associatedMeanings: associatedMeanings ?? self.associatedMeanings,
            contributingSegments: contributingSegments ?? self.contributingSegments,
            createdAt: createdAt,
            lastUpdated: lastUpdated
        )
    }

    // Ignore timestamps and tiny float jitter
    func meaningfullyDiffers(from other: CompositionalPattern) -> Bool {
        @inline(__always) func approx(_ a: Float, _ b: Float, tol: Float = 1e-4) -> Bool { abs(a - b) <= tol }
        @inline(__always) func approxArray(_ a: [Float], _ b: [Float]) -> Bool {
            guard a.count == b.count else { return false }
            for i in 0..<a.count { if !approx(a[i], b[i]) { return true } }
            return false
        }
        if id != other.id { return true }
        if approxArray(representativeEmbedding, other.representativeEmbedding) { } else { return true }
        if frequency != other.frequency { return true }
        if !approx(confidence, other.confidence) { return true }
        if !approx(averagePosition, other.averagePosition) { return true }
        if associatedMeanings != other.associatedMeanings { return true }
        if contributingSegments != other.contributingSegments { return true }
        return false
    }

    // Private memberwise used by with(...)
    private init(
        id: UUID,
        representativeEmbedding: [Float],
        frequency: Int,
        confidence: Float,
        averagePosition: Float,
        associatedMeanings: [String],
        contributingSegments: [UUID],
        createdAt: Date,
        lastUpdated: Date
    ) {
        self.id = id
        self.representativeEmbedding = representativeEmbedding
        self.frequency = frequency
        self.confidence = confidence
        self.averagePosition = averagePosition
        self.associatedMeanings = associatedMeanings
        self.contributingSegments = contributingSegments
        self.createdAt = createdAt
        self.lastUpdated = lastUpdated
    }
}

// ============================================================
// MARK: - PatternMatchResult
// ============================================================

struct MatchedPatternInfo: Codable {
    let pattern: CompositionalPattern
    let confidence: Float
    let position: TimeInterval
}

struct PatternMatchResult: Codable {
    let matchedPatterns: [MatchedPatternInfo]
    let overallConfidence: Float
    let reconstructedTranslation: String
    let explanation: String
    
    var isSuccessful: Bool { overallConfidence > 0.0 }
    var hasMatches: Bool { !matchedPatterns.isEmpty || overallConfidence > 0.0 }
}

// ============================================================
// MARK: - PatternCollection
// ============================================================

struct PatternCollection: Codable {
    private(set) var patterns: [CompositionalPattern] = []
    private(set) var lastDiscoveryRun: Date? = nil

    var significantPatterns: [CompositionalPattern] {
        patterns.filter { $0.isSignificant }
    }

    mutating func addPattern(_ p: CompositionalPattern) { upsert(p) }

    mutating func upsert(_ p: CompositionalPattern) {
        if let i = patterns.firstIndex(where: { $0.id == p.id }) {
            patterns[i] = p
        } else {
            patterns.append(p)
        }
    }

    mutating func updatePattern(id: UUID, newSegments: [AudioSegment], newMeanings: [String]) {
        guard let idx = patterns.firstIndex(where: { $0.id == id }) else { return }
        let cur = patterns[idx]

        let mergedMeanings = Array(Set(cur.associatedMeanings + newMeanings)).sorted()
        let mergedSegIds = cur.contributingSegments + newSegments.map(\.id)
        let newFreq = cur.frequency + newSegments.count

        // Running average of averagePosition using segment starts
        let addedPosSum = newSegments.map { Float($0.startTime) }.reduce(0, +)
        let oldWeighted = cur.averagePosition * Float(max(1, cur.frequency))
        let newAvgPos = (oldWeighted + addedPosSum) / Float(max(1, newFreq))

        // Simple confidence reinforcement
        let newConf = min(1.0, cur.confidence + Float(newSegments.count) * 0.1)

        let updated = cur.with(
            frequency: newFreq,
            confidence: newConf,
            averagePosition: newAvgPos,
            associatedMeanings: mergedMeanings,
            contributingSegments: mergedSegIds,
            lastUpdated: Date()
        )

        if updated.meaningfullyDiffers(from: cur) {
            patterns[idx] = updated
        }
    }

    mutating func removeWeakPatterns() {
        patterns.removeAll { !$0.isSignificant }
    }
    
    mutating func removeWeakPatterns(segmentsById: [UUID: AudioSegment],
                                   examplesById: [UUID: TrainingExample],
                                   cfg: PatternDiscoveryConfig) {
        let initialCount = patterns.count
        patterns.removeAll { pattern in
            !PatternValidator.isValid(pattern, 
                                    segmentsById: segmentsById,
                                    examplesById: examplesById,
                                    cfg: cfg)
        }
        let removedCount = initialCount - patterns.count
        
        #if DEBUG
        if removedCount > 0 {
            let removalPercentage = Float(removedCount) / Float(initialCount) * 100
            print("🧹 Pruned \(removedCount)/\(initialCount) patterns (\(String(format: "%.1f", removalPercentage))%)")
        }
        #endif
    }

    mutating func markDiscoveryComplete() { lastDiscoveryRun = Date() }
    
    /// Aggressively prune patterns to keep memory usage low
    mutating func aggressivePrune() {
        let initialCount = patterns.count
        
        // Remove patterns with low frequency/confidence first
        patterns.removeAll { pattern in
            pattern.frequency < 3 || pattern.confidence < 0.6
        }
        
        // If still too many patterns, keep only the most significant ones
        if patterns.count > 8 {
            patterns = Array(patterns
                .sorted { $0.confidence > $1.confidence }
                .prefix(8))
        }
        
        let removedCount = initialCount - patterns.count
        if removedCount > 0 {
            print("🧹 Aggressive pruning removed \(removedCount) patterns, kept \(patterns.count)")
        }
    }

    func findSimilarPatterns(to embedding: [Float], threshold: Float = 0.7)
      -> [(pattern: CompositionalPattern, similarity: Float)] {
        patterns.compactMap { p in
            let sim = cosineSimilarity(embedding, p.representativeEmbedding)
            return sim > threshold ? (p, sim) : nil
        }
        .sorted { $0.similarity > $1.similarity }
    }
}

// ============================================================
// MARK: - Discovery Config
// ============================================================

enum SegmentationStrategy: Codable {
    case fixed(window: TimeInterval, overlap: Double)
    case variable(minDuration: TimeInterval, maxDuration: TimeInterval, overlap: Double)
    case adaptive
    case embeddingBased(minDuration: TimeInterval, maxDuration: TimeInterval, similarityThreshold: Float)
}

struct PatternDiscoveryConfig: Codable {
    let segmentationStrategy: SegmentationStrategy
    let similarityThreshold: Float
    let minPatternFrequency: Int
    let maxPatternsToDiscover: Int
    let minPatternConfidence: Float
    let meaningConsistencyThreshold: Float

    static let `default` = PatternDiscoveryConfig(
        segmentationStrategy: .embeddingBased(minDuration: 0.2, maxDuration: 2.0, similarityThreshold: 0.6),
        // Relaxed config for testing - easier discovery with limited data
        similarityThreshold: 0.5,
        minPatternFrequency: 1,
        maxPatternsToDiscover: 20,
        minPatternConfidence: 0.25,
        meaningConsistencyThreshold: 0.6
    )
}

// ============================================================
// MARK: - Utilities
// ============================================================

@inline(__always)
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot: Float = 0, aa: Float = 0, bb: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        aa += a[i] * a[i]
        bb += b[i] * b[i]
    }
    let denom = sqrt(aa) * sqrt(bb)
    let result = denom > 1e-6 ? dot / denom : 0
    return result.isFinite ? result : 0
}

// NaN safety utilities
extension CGFloat {
    var finiteOrZero: CGFloat {
        isFinite ? self : 0
    }
}

@inline(__always)
func safeDiv(_ a: CGFloat, _ b: CGFloat) -> CGFloat {
    let result = a / max(b, 1e-6)
    return result.finiteOrZero
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
