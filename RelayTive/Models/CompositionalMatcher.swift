//
//  CompositionalMatcher.swift
//  RelayTive
//
//  Engine for matching new audio against discovered compositional patterns
//

import Foundation

/// Engine responsible for matching new audio against learned compositional patterns
class CompositionalMatcher {
    private let segmentationEngine: AudioSegmentationEngine
    private let config: CompositionalMatchingConfig
    
    init(segmentationEngine: AudioSegmentationEngine, config: CompositionalMatchingConfig = .default) {
        self.segmentationEngine = segmentationEngine
        self.config = config
    }
    
    // MARK: - Main Matching Method
    
    /// Match new audio against discovered patterns and attempt to reconstruct translation
    func matchAudio(_ audioData: Data, 
                   against patternCollection: PatternCollection,
                   fallbackExamples: [TrainingExample]) async -> PatternMatchResult {
        
        print("🎯 Matching new audio against \(patternCollection.significantPatterns.count) patterns")
        
        // Create temporary training example for segmentation
        let tempExample = TrainingExample(
            atypicalAudio: audioData,
            typicalExplanation: "",
            timestamp: Date()
        )
        
        // Step 1: Segment the new audio
        let segments = await segmentationEngine.extractSegments(from: tempExample, 
                                                               strategy: config.segmentationStrategy)
        print("📊 Extracted \(segments.count) segments from new audio")
        
        // Step 2: Match each segment against patterns
        let segmentMatches = matchSegmentsToPatterns(segments, patterns: patternCollection.significantPatterns)
        print("🔍 Found \(segmentMatches.count) segment-to-pattern matches")
        
        // Step 3: Attempt compositional translation reconstruction
        if let compositionalResult = reconstructCompositionalTranslation(from: segmentMatches, segments: segments) {
            print("✅ Successfully reconstructed compositional translation: \(compositionalResult.reconstructedTranslation)")
            return compositionalResult
        }
        
        // Step 4: Fallback to whole-utterance matching
        print("⚠️ Compositional matching insufficient, falling back to whole-utterance matching")
        return await fallbackToWholeUtteranceMatching(audioData, segments: segments, examples: fallbackExamples)
    }
    
    // MARK: - Segment-to-Pattern Matching
    
    private func matchSegmentsToPatterns(_ segments: [AudioSegment], 
                                       patterns: [CompositionalPattern]) -> [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)] {
        var matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)] = []
        
        for segment in segments {
            var bestMatch: (pattern: CompositionalPattern, confidence: Float)?
            
            for pattern in patterns {
                let similarity = cosineSimilarity(segment.embeddings, pattern.representativeEmbedding)
                
                if similarity > config.minMatchConfidence &&
                   (bestMatch == nil || similarity > bestMatch!.confidence) {
                    bestMatch = (pattern, similarity)
                }
            }
            
            if let match = bestMatch {
                matches.append((segment: segment, pattern: match.pattern, confidence: match.confidence))
                print("  🎯 Segment at \(segment.startTime)s matched pattern (freq: \(match.pattern.frequency)) with confidence \(match.confidence)")
            }
        }
        
        return matches.sorted { $0.segment.startTime < $1.segment.startTime }
    }
    
    // MARK: - Compositional Translation Reconstruction
    
    private func reconstructCompositionalTranslation(from matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)], 
                                                   segments: [AudioSegment]) -> PatternMatchResult? {
        
        guard !matches.isEmpty else { return nil }
        
        // Calculate coverage of the audio by matched patterns
        let coverage = calculatePatternCoverage(matches: matches, totalSegments: segments)
        
        guard coverage >= config.minCoverageThreshold else {
            print("  📊 Pattern coverage too low: \(coverage) < \(config.minCoverageThreshold)")
            return nil
        }
        
        // Strategy 1: Try meaning combination approach
        if let meaningBasedResult = reconstructFromMeaningCombination(matches: matches, coverage: coverage) {
            return meaningBasedResult
        }
        
        // Strategy 2: Try frequency-weighted selection
        if let frequencyBasedResult = reconstructFromFrequencyWeighting(matches: matches, coverage: coverage) {
            return frequencyBasedResult
        }
        
        // Strategy 3: Try dominant pattern approach
        return reconstructFromDominantPattern(matches: matches, coverage: coverage)
    }
    
    private func reconstructFromMeaningCombination(matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)], 
                                                 coverage: Float) -> PatternMatchResult? {
        
        print("🔧 Attempting meaning combination reconstruction")
        
        // Collect all meanings from matched patterns, weighted by confidence
        var meaningWeights: [String: Float] = [:]
        var patternDetails: [(pattern: CompositionalPattern, confidence: Float, position: TimeInterval)] = []
        
        for match in matches {
            let weight = match.confidence * match.confidence // Square for emphasis on high-confidence matches
            
            for meaning in match.pattern.associatedMeanings {
                meaningWeights[meaning, default: 0] += weight
            }
            
            patternDetails.append((
                pattern: match.pattern,
                confidence: match.confidence,
                position: match.segment.startTime
            ))
        }
        
        // Find the most weighted meaning
        guard let (bestMeaning, bestWeight) = meaningWeights.max(by: { $0.value < $1.value }),
              bestWeight > config.minCombinedConfidence else {
            print("  ❌ No meaning combination meets confidence threshold")
            return nil
        }
        
        let overallConfidence = min(1.0, bestWeight * coverage)
        let explanation = "Compositional match using meaning combination (coverage: \(Int(coverage * 100))%)"
        
        return PatternMatchResult(
            matchedPatterns: patternDetails,
            overallConfidence: overallConfidence,
            reconstructedTranslation: bestMeaning,
            explanation: explanation
        )
    }
    
    private func reconstructFromFrequencyWeighting(matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)], 
                                                 coverage: Float) -> PatternMatchResult? {
        
        print("🔧 Attempting frequency-weighted reconstruction")
        
        // Weight patterns by both confidence and frequency
        var weightedMeanings: [String: Float] = [:]
        var patternDetails: [(pattern: CompositionalPattern, confidence: Float, position: TimeInterval)] = []
        
        for match in matches {
            let frequencyWeight = min(1.0, Float(match.pattern.frequency) / 10.0) // Cap frequency influence
            let combinedWeight = match.confidence * frequencyWeight
            
            for meaning in match.pattern.associatedMeanings {
                weightedMeanings[meaning, default: 0] += combinedWeight
            }
            
            patternDetails.append((
                pattern: match.pattern,
                confidence: match.confidence,
                position: match.segment.startTime
            ))
        }
        
        guard let (bestMeaning, bestWeight) = weightedMeanings.max(by: { $0.value < $1.value }),
              bestWeight > config.minCombinedConfidence else {
            print("  ❌ No frequency-weighted combination meets confidence threshold")
            return nil
        }
        
        let overallConfidence = min(1.0, bestWeight * coverage)
        let explanation = "Compositional match using frequency weighting (coverage: \(Int(coverage * 100))%)"
        
        return PatternMatchResult(
            matchedPatterns: patternDetails,
            overallConfidence: overallConfidence,
            reconstructedTranslation: bestMeaning,
            explanation: explanation
        )
    }
    
    private func reconstructFromDominantPattern(matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)], 
                                              coverage: Float) -> PatternMatchResult? {
        
        print("🔧 Attempting dominant pattern reconstruction")
        
        // Find the pattern with highest confidence * frequency product
        guard let dominantMatch = matches.max(by: { 
            ($0.confidence * Float($0.pattern.frequency)) < ($1.confidence * Float($1.pattern.frequency)) 
        }) else { return nil }
        
        // Use the most common meaning from the dominant pattern
        let meaningCounts = Dictionary(dominantMatch.pattern.associatedMeanings.map { ($0, 1) }, uniquingKeysWith: +)
        guard let (dominantMeaning, _) = meaningCounts.max(by: { $0.value < $1.value }) else { return nil }
        
        let patternDetails = matches.map { match in
            (pattern: match.pattern, confidence: match.confidence, position: match.segment.startTime)
        }
        
        let overallConfidence = min(1.0, dominantMatch.confidence * coverage)
        let explanation = "Compositional match using dominant pattern approach (coverage: \(Int(coverage * 100))%)"
        
        guard overallConfidence > config.minCombinedConfidence else {
            print("  ❌ Dominant pattern confidence too low: \(overallConfidence)")
            return nil
        }
        
        return PatternMatchResult(
            matchedPatterns: patternDetails,
            overallConfidence: overallConfidence,
            reconstructedTranslation: dominantMeaning,
            explanation: explanation
        )
    }
    
    // MARK: - Fallback Matching
    
    private func fallbackToWholeUtteranceMatching(_ audioData: Data, 
                                                segments: [AudioSegment], 
                                                examples: [TrainingExample]) async -> PatternMatchResult {
        
        print("🔄 Performing whole-utterance fallback matching against \(examples.count) training examples")
        
        // Create a temporary training example for the new audio
        let tempExample = TrainingExample(atypicalAudio: audioData, typicalExplanation: "", timestamp: Date())
        
        // Try to extract whole-utterance embeddings
        guard let wholeUtteranceEmbeddings = tempExample.audioEmbeddings else {
            print("  ❌ No embeddings available for whole-utterance matching")
            return createNoMatchResult()
        }
        
        // Find best matching training example
        var bestMatch: (example: TrainingExample, similarity: Float)?
        
        for example in examples {
            guard let exampleEmbeddings = example.audioEmbeddings else { continue }
            
            let similarity = cosineSimilarity(wholeUtteranceEmbeddings, exampleEmbeddings)
            
            if similarity > config.fallbackSimilarityThreshold &&
               (bestMatch == nil || similarity > bestMatch!.similarity) {
                bestMatch = (example, similarity)
            }
        }
        
        if let match = bestMatch {
            print("  ✅ Found whole-utterance match with \(match.similarity) similarity")
            
            return PatternMatchResult(
                matchedPatterns: [], // No compositional patterns
                overallConfidence: match.similarity,
                reconstructedTranslation: match.example.typicalExplanation,
                explanation: "Whole-utterance match (similarity: \(Int(match.similarity * 100))%)"
            )
        } else {
            print("  ❌ No suitable whole-utterance match found")
            return createNoMatchResult()
        }
    }
    
    // MARK: - Utility Methods
    
    private func calculatePatternCoverage(matches: [(segment: AudioSegment, pattern: CompositionalPattern, confidence: Float)], 
                                        totalSegments: [AudioSegment]) -> Float {
        guard !totalSegments.isEmpty else { return 0.0 }
        
        // Calculate temporal coverage
        let totalDuration = totalSegments.max { $0.endTime < $1.endTime }?.endTime ?? 0
        let matchedDuration = matches.map { $0.segment.duration }.reduce(0, +)
        
        let temporalCoverage = totalDuration > 0 ? Float(matchedDuration / totalDuration) : 0
        
        // Calculate segment coverage  
        let segmentCoverage = Float(matches.count) / Float(totalSegments.count)
        
        // Combined coverage (weighted average)
        return (temporalCoverage * 0.6) + (segmentCoverage * 0.4)
    }
    
    private func createNoMatchResult() -> PatternMatchResult {
        return PatternMatchResult(
            matchedPatterns: [],
            overallConfidence: 0.0,
            reconstructedTranslation: "No matching pattern found. Please add more training examples.",
            explanation: "No compositional or whole-utterance matches found"
        )
    }
}

// MARK: - Configuration

struct CompositionalMatchingConfig {
    let minMatchConfidence: Float           // Minimum confidence for segment-to-pattern matching
    let minCoverageThreshold: Float         // Minimum coverage required for compositional reconstruction
    let minCombinedConfidence: Float        // Minimum combined confidence for accepting reconstructed translation
    let fallbackSimilarityThreshold: Float // Similarity threshold for whole-utterance fallback
    let segmentationStrategy: SegmentationStrategy // Strategy for segmenting new audio
    
    static let `default` = CompositionalMatchingConfig(
        minMatchConfidence: 0.6,
        minCoverageThreshold: 0.4,
        minCombinedConfidence: 0.5,
        fallbackSimilarityThreshold: 0.7,
        segmentationStrategy: .variable(minDuration: 0.2, maxDuration: 1.0)
    )
    
    static let conservative = CompositionalMatchingConfig(
        minMatchConfidence: 0.75,
        minCoverageThreshold: 0.6,
        minCombinedConfidence: 0.7,
        fallbackSimilarityThreshold: 0.8,
        segmentationStrategy: .variable(minDuration: 0.3, maxDuration: 0.8)
    )
    
    static let aggressive = CompositionalMatchingConfig(
        minMatchConfidence: 0.5,
        minCoverageThreshold: 0.3,
        minCombinedConfidence: 0.4,
        fallbackSimilarityThreshold: 0.6,
        segmentationStrategy: .variable(minDuration: 0.1, maxDuration: 1.2)
    )
}

// MARK: - Matching Analytics

extension CompositionalMatcher {
    
    /// Analyze the quality of pattern matches for debugging and optimization
    func analyzeMatchQuality(_ result: PatternMatchResult) -> MatchQualityAnalysis {
        let patternCount = result.matchedPatterns.count
        let averagePatternConfidence = patternCount > 0 ? 
            result.matchedPatterns.map { $0.confidence }.reduce(0, +) / Float(patternCount) : 0
        
        let temporalDistribution = analyzeTemporalDistribution(result.matchedPatterns)
        let confidenceDistribution = analyzeConfidenceDistribution(result.matchedPatterns)
        
        return MatchQualityAnalysis(
            patternCount: patternCount,
            overallConfidence: result.overallConfidence,
            averagePatternConfidence: averagePatternConfidence,
            temporalDistribution: temporalDistribution,
            confidenceDistribution: confidenceDistribution,
            explanation: result.explanation
        )
    }
    
    private func analyzeTemporalDistribution(_ matches: [(pattern: CompositionalPattern, confidence: Float, position: TimeInterval)]) -> String {
        guard !matches.isEmpty else { return "No matches" }
        
        let positions = matches.map { $0.position }
        let minPos = positions.min() ?? 0
        let maxPos = positions.max() ?? 0
        let span = maxPos - minPos
        
        return span > 1.0 ? "Well distributed" : "Clustered"
    }
    
    private func analyzeConfidenceDistribution(_ matches: [(pattern: CompositionalPattern, confidence: Float, position: TimeInterval)]) -> String {
        guard !matches.isEmpty else { return "No matches" }
        
        let confidences = matches.map { $0.confidence }
        let avgConfidence = confidences.reduce(0, +) / Float(confidences.count)
        let variance = confidences.map { pow($0 - avgConfidence, 2) }.reduce(0, +) / Float(confidences.count)
        
        return variance < 0.1 ? "Consistent" : "Variable"
    }
}

struct MatchQualityAnalysis {
    let patternCount: Int
    let overallConfidence: Float
    let averagePatternConfidence: Float
    let temporalDistribution: String
    let confidenceDistribution: String
    let explanation: String
    
    var qualityScore: Float {
        let countFactor = min(1.0, Float(patternCount) / 5.0)
        let confidenceFactor = (overallConfidence + averagePatternConfidence) / 2.0
        return (countFactor * 0.3) + (confidenceFactor * 0.7)
    }
}