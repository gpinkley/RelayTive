//
//  PatternDiscoveryEngine.swift
//  RelayTive
//
//  Engine for discovering recurring compositional patterns across training examples
//

import Foundation

/// Engine responsible for discovering compositional patterns from audio segments
class PatternDiscoveryEngine {
    private let config: PatternDiscoveryConfig
    private let segmentationEngine: AudioSegmentationEngine
    
    init(config: PatternDiscoveryConfig = .default, segmentationEngine: AudioSegmentationEngine) {
        self.config = config
        self.segmentationEngine = segmentationEngine
    }
    
    // MARK: - Main Discovery Method
    
    /// Discover compositional patterns from a collection of training examples
    func discoverPatterns(from trainingExamples: [TrainingExample]) async -> PatternCollection {
        print("🔍 Starting pattern discovery from \(trainingExamples.count) training examples")
        
        var patternCollection = PatternCollection()
        
        // Step 1: Extract segments from all training examples
        let allSegments = await extractAllSegments(from: trainingExamples)
        print("📊 Extracted \(allSegments.count) total segments for pattern analysis")
        
        // Step 2: Cluster similar segments to find recurring patterns
        let clusters = clusterSimilarSegments(allSegments)
        print("🎯 Found \(clusters.count) segment clusters")
        
        // Step 3: Convert significant clusters to compositional patterns
        for cluster in clusters {
            if let pattern = createPatternFromCluster(cluster, allTrainingExamples: trainingExamples) {
                patternCollection.addPattern(pattern)
            }
        }
        
        // Step 4: Clean up and optimize patterns
        patternCollection.removeWeakPatterns()
        patternCollection.markDiscoveryComplete()
        
        print("✅ Pattern discovery complete: \(patternCollection.significantPatterns.count) significant patterns found")
        return patternCollection
    }
    
    // MARK: - Segment Extraction
    
    private func extractAllSegments(from trainingExamples: [TrainingExample]) async -> [AudioSegment] {
        var allSegments: [AudioSegment] = []
        
        for (index, example) in trainingExamples.enumerated() {
            print("⏱️ Processing training example \(index + 1)/\(trainingExamples.count): \(example.typicalExplanation)")
            
            let segments = await segmentationEngine.extractSegments(from: example, strategy: config.segmentationStrategy)
            allSegments.append(contentsOf: segments)
            
            print("  📈 Added \(segments.count) segments (total: \(allSegments.count))")
        }
        
        return allSegments.filter { $0.isValid }
    }
    
    // MARK: - Clustering Algorithm
    
    private func clusterSimilarSegments(_ segments: [AudioSegment]) -> [[AudioSegment]] {
        print("🔬 Clustering \(segments.count) segments using similarity threshold \(config.similarityThreshold)")
        
        var clusters: [[AudioSegment]] = []
        var unclusteredSegments = segments
        
        while !unclusteredSegments.isEmpty {
            let seed = unclusteredSegments.removeFirst()
            var currentCluster = [seed]
            
            // Find all segments similar to the seed
            var remainingSegments: [AudioSegment] = []
            
            for segment in unclusteredSegments {
                let similarity = cosineSimilarity(seed.embeddings, segment.embeddings)
                
                if similarity > config.similarityThreshold {
                    currentCluster.append(segment)
                } else {
                    remainingSegments.append(segment)
                }
            }
            
            unclusteredSegments = remainingSegments
            
            // Only keep clusters that meet minimum frequency requirement
            if currentCluster.count >= config.minPatternFrequency {
                clusters.append(currentCluster)
                print("  🎯 Created cluster with \(currentCluster.count) segments (similarity > \(config.similarityThreshold))")
            }
        }
        
        // Sort clusters by size (frequency) descending
        clusters.sort { $0.count > $1.count }
        
        // Limit to maximum patterns if specified
        if clusters.count > config.maxPatternsToDiscover {
            clusters = Array(clusters.prefix(config.maxPatternsToDiscover))
            print("  📏 Limited to top \(config.maxPatternsToDiscover) clusters")
        }
        
        return clusters
    }
    
    // MARK: - Pattern Creation
    
    private func createPatternFromCluster(_ cluster: [AudioSegment], allTrainingExamples: [TrainingExample]) -> CompositionalPattern? {
        guard cluster.count >= config.minPatternFrequency else { return nil }
        
        print("🏗️ Creating pattern from cluster of \(cluster.count) segments")
        
        // Calculate representative embedding as average of cluster embeddings
        let clusterEmbeddings = cluster.map { $0.embeddings }
        guard let representativeEmbedding = averageEmbedding(from: clusterEmbeddings) else {
            print("❌ Failed to calculate representative embedding for cluster")
            return nil
        }
        
        // Find associated meanings from training examples
        let associatedMeanings = findAssociatedMeanings(for: cluster, in: allTrainingExamples)
        
        // Calculate pattern quality metrics
        let averageConfidence = cluster.map { $0.confidence }.reduce(0, +) / Float(cluster.count)
        let positionVariance = calculatePositionVariance(for: cluster)
        
        // Adjust confidence based on consistency
        let consistencyFactor = max(0.1, 1.0 - positionVariance) // Lower variance = higher consistency
        let adjustedConfidence = min(1.0, averageConfidence * consistencyFactor * Float(cluster.count) * 0.1)
        
        guard adjustedConfidence >= config.minPatternConfidence else {
            print("  ⚠️ Pattern confidence too low: \(adjustedConfidence) < \(config.minPatternConfidence)")
            return nil
        }
        
        let pattern = CompositionalPattern(
            representativeEmbedding: representativeEmbedding,
            segments: cluster,
            associatedMeanings: associatedMeanings
        )
        
        print("  ✅ Created pattern: frequency=\(pattern.frequency), confidence=\(pattern.confidence), meanings=\(associatedMeanings.count)")
        
        return pattern
    }
    
    private func findAssociatedMeanings(for cluster: [AudioSegment], in trainingExamples: [TrainingExample]) -> [String] {
        var meanings: [String] = []
        
        // Group segments by parent example
        let segmentsByExample = Dictionary(grouping: cluster) { $0.parentExampleId }
        
        for (exampleId, _) in segmentsByExample {
            if let example = trainingExamples.first(where: { $0.id == exampleId }) {
                meanings.append(example.typicalExplanation)
            }
        }
        
        // Remove duplicates and return unique meanings
        return Array(Set(meanings))
    }
    
    private func calculatePositionVariance(for cluster: [AudioSegment]) -> Float {
        let positions = cluster.map { Float($0.startTime) }
        let meanPosition = positions.reduce(0, +) / Float(positions.count)
        
        let variance = positions.map { pow($0 - meanPosition, 2) }.reduce(0, +) / Float(positions.count)
        return sqrt(variance)
    }
    
    // MARK: - Pattern Validation
    
    /// Validate discovered patterns against new training data
    func validatePatterns(_ patterns: PatternCollection, against newTrainingExamples: [TrainingExample]) async -> PatternCollection {
        print("🔍 Validating \(patterns.patterns.count) patterns against \(newTrainingExamples.count) new examples")
        
        var updatedCollection = patterns
        
        // Extract segments from new examples
        let newSegments = await extractAllSegments(from: newTrainingExamples)
        
        for pattern in patterns.patterns {
            // Find segments that match this pattern
            let matchingSegments = newSegments.filter { segment in
                cosineSimilarity(segment.embeddings, pattern.representativeEmbedding) > config.similarityThreshold
            }
            
            if !matchingSegments.isEmpty {
                // Update pattern with new evidence
                let newMeanings = findAssociatedMeanings(for: matchingSegments, in: newTrainingExamples)
                updatedCollection.updatePattern(id: pattern.id, newSegments: matchingSegments, newMeanings: newMeanings)
                
                print("  📈 Updated pattern \(pattern.id) with \(matchingSegments.count) new matching segments")
            }
        }
        
        return updatedCollection
    }
    
    // MARK: - Incremental Discovery
    
    /// Add new segments to existing pattern collection
    func updatePatterns(_ patterns: PatternCollection, with newTrainingExamples: [TrainingExample]) async -> PatternCollection {
        print("🔄 Updating pattern collection with \(newTrainingExamples.count) new training examples")
        
        var updatedCollection = patterns
        
        // Extract segments from new examples
        let newSegments = await extractAllSegments(from: newTrainingExamples)
        
        // Check if new segments fit existing patterns
        for segment in newSegments {
            let matchingPatterns = updatedCollection.findSimilarPatterns(to: segment.embeddings, threshold: config.similarityThreshold)
            
            if matchingPatterns.isEmpty {
                // No existing pattern matches - this could seed a new pattern
                // For now, we'll require manual pattern discovery runs for new patterns
                continue
            } else {
                // Segment matches existing pattern - update the pattern
                if let bestMatch = matchingPatterns.first {
                    let newMeanings = findAssociatedMeanings(for: [segment], in: newTrainingExamples)
                    updatedCollection.updatePattern(id: bestMatch.pattern.id, newSegments: [segment], newMeanings: newMeanings)
                }
            }
        }
        
        return updatedCollection
    }
}

// MARK: - Pattern Analysis Utilities

extension PatternDiscoveryEngine {
    
    /// Analyze pattern quality and suggest improvements
    func analyzePatternQuality(_ patterns: PatternCollection) -> PatternQualityReport {
        let totalPatterns = patterns.patterns.count
        let significantPatterns = patterns.significantPatterns.count
        
        let averageFrequency = patterns.patterns.isEmpty ? 0 : 
            patterns.patterns.map { $0.frequency }.reduce(0, +) / patterns.patterns.count
        
        let averageConfidence = patterns.patterns.isEmpty ? 0 : 
            patterns.patterns.map { $0.confidence }.reduce(0, +) / Float(patterns.patterns.count)
        
        let coverageAnalysis = analyzeCoverage(patterns, totalPatterns: totalPatterns)
        
        return PatternQualityReport(
            totalPatterns: totalPatterns,
            significantPatterns: significantPatterns,
            averageFrequency: averageFrequency,
            averageConfidence: averageConfidence,
            coverage: coverageAnalysis,
            recommendations: generateRecommendations(patterns)
        )
    }
    
    private func analyzeCoverage(_ patterns: PatternCollection, totalPatterns: Int) -> Float {
        // Simple coverage metric: ratio of significant to total patterns
        return totalPatterns > 0 ? Float(patterns.significantPatterns.count) / Float(totalPatterns) : 0
    }
    
    private func generateRecommendations(_ patterns: PatternCollection) -> [String] {
        var recommendations: [String] = []
        
        if patterns.significantPatterns.count < 3 {
            recommendations.append("Consider adding more diverse training examples to discover additional patterns")
        }
        
        let lowConfidencePatterns = patterns.patterns.filter { $0.confidence < 0.5 }.count
        if lowConfidencePatterns > patterns.patterns.count / 2 {
            recommendations.append("Many patterns have low confidence - consider adjusting similarity thresholds")
        }
        
        let lowFrequencyPatterns = patterns.patterns.filter { $0.frequency < 3 }.count
        if lowFrequencyPatterns > patterns.patterns.count / 3 {
            recommendations.append("Consider increasing minimum pattern frequency to reduce noise")
        }
        
        return recommendations
    }
}

// MARK: - Supporting Types

struct PatternQualityReport {
    let totalPatterns: Int
    let significantPatterns: Int
    let averageFrequency: Int
    let averageConfidence: Float
    let coverage: Float
    let recommendations: [String]
    
    var qualityScore: Float {
        return (coverage * 0.4) + (averageConfidence * 0.6)
    }
}