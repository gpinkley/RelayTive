//
//  PatternDiscoveryEngine.swift
//  RelayTive
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

    // MARK: - Main Discovery

    func discoverPatterns(from trainingExamples: [TrainingExample]) async -> PatternCollection {
        print("🔍 Starting pattern discovery from \(trainingExamples.count) training examples")
        var collection = PatternCollection()

        let allSegments = await extractAllSegments(from: trainingExamples)
        print("📊 Extracted \(allSegments.count) total segments")

        let clusters = clusterSimilarSegments(allSegments)
        print("🎯 Found \(clusters.count) segment clusters")

        for cluster in clusters {
            if let pattern = createPatternFromCluster(cluster, allTrainingExamples: trainingExamples) {
                collection.addPattern(pattern)
            }
        }

        // Create lookup maps for enhanced pattern validation
        let segmentsById = Dictionary(uniqueKeysWithValues: allSegments.map { ($0.id, $0) })
        let examplesById = Dictionary(uniqueKeysWithValues: trainingExamples.map { ($0.id, $0) })
        
        collection.removeWeakPatterns(segmentsById: segmentsById, 
                                    examplesById: examplesById, 
                                    cfg: config)
        
        #if DEBUG
        PatternDebug.debugReportPatterns(collection.patterns,
                                       segmentsById: segmentsById,
                                       examplesById: examplesById,
                                       cfg: config)
        #endif
        
        collection.markDiscoveryComplete()
        print("✅ Pattern discovery complete: \(collection.significantPatterns.count) significant patterns")
        return collection
    }
    
    /// Update existing patterns with new training examples
    public func updatePatterns(_ existingPatterns: PatternCollection, with newExamples: [TrainingExample]) async -> PatternCollection {
        print("🔄 Updating patterns with \(newExamples.count) new examples")
        var updatedCollection = existingPatterns
        
        // Extract segments from new examples
        let newSegments = await extractAllSegments(from: newExamples)
        print("📊 Extracted \(newSegments.count) segments from new examples")
        
        // For each new segment, try to match it to existing patterns
        for segment in newSegments {
            var bestMatch: (pattern: CompositionalPattern, similarity: Float)?
            
            // Find the best matching existing pattern
            for pattern in updatedCollection.significantPatterns {
                let similarity = cosineSimilarity(segment.embeddings, pattern.representativeEmbedding)
                
                if similarity > config.similarityThreshold &&
                   (bestMatch == nil || similarity > bestMatch!.similarity) {
                    bestMatch = (pattern, similarity)
                }
            }
            
            // If we found a good match, update the pattern
            if let match = bestMatch {
                // Find the associated meaning for this segment
                let associatedMeaning = newExamples.first { $0.id == segment.parentExampleId }?.typicalExplanation ?? ""
                let newMeanings = associatedMeaning.isEmpty ? [] : [associatedMeaning]
                
                updatedCollection.updatePattern(id: match.pattern.id, 
                                              newSegments: [segment], 
                                              newMeanings: newMeanings)
                print("  🎯 Updated pattern \(match.pattern.id) with new segment (similarity: \(match.similarity))")
            } else {
                // No good match found - this could seed a new pattern
                // For now, we'll skip creating new patterns from single segments
                print("  ⚠️ No matching pattern found for new segment")
            }
        }
        
        // Run a mini discovery on remaining unmatched segments to potentially create new patterns
        let unmatchedSegments = newSegments.filter { segment in
            !updatedCollection.significantPatterns.contains { pattern in
                cosineSimilarity(segment.embeddings, pattern.representativeEmbedding) > config.similarityThreshold
            }
        }
        
        if unmatchedSegments.count >= config.minPatternFrequency {
            print("🔍 Running mini-discovery on \(unmatchedSegments.count) unmatched segments")
            let clusters = clusterSimilarSegments(unmatchedSegments)
            
            for cluster in clusters {
                if let newPattern = createPatternFromCluster(cluster, allTrainingExamples: newExamples) {
                    updatedCollection.addPattern(newPattern)
                    print("  ✅ Created new pattern from unmatched segments")
                }
            }
        }
        
        // Clean up weak patterns
        updatedCollection.removeWeakPatterns()
        updatedCollection.markDiscoveryComplete()
        
        print("✅ Pattern update complete: \(updatedCollection.significantPatterns.count) significant patterns")
        return updatedCollection
    }

    // MARK: - Segment Extraction

    private func extractAllSegments(from trainingExamples: [TrainingExample]) async -> [AudioSegment] {
        var out: [AudioSegment] = []
        let limited = Array(trainingExamples.prefix(10))

        for (i, ex) in limited.enumerated() {
            print("⏱️ Processing example \(i + 1)/\(limited.count): \(ex.typicalExplanation)")
            let segs = await segmentationEngine.extractSegments(from: ex, strategy: config.segmentationStrategy)
            out.append(contentsOf: segs)
            print("  📈 +\(segs.count) segments (total \(out.count))")
            if out.count > 50 {
                print("  ⚠️ Segment cap hit, stopping")
                break
            }
        }
        return out.filter { $0.isValid }
    }

    // MARK: - Clustering

    private func clusterSimilarSegments(_ segments: [AudioSegment]) -> [[AudioSegment]] {
        print("🔬 Clustering \(segments.count) segments, threshold \(config.similarityThreshold)")

        var clusters: [[AudioSegment]] = []
        var remaining = segments

        while !remaining.isEmpty {
            let seed = remaining.removeFirst()
            var cluster = [seed]
            var keep: [AudioSegment] = []

            for s in remaining {
                let sim = cosineSimilarity(seed.embeddings, s.embeddings)
                if sim > config.similarityThreshold { cluster.append(s) } else { keep.append(s) }
            }
            remaining = keep

            if cluster.count >= config.minPatternFrequency {
                clusters.append(cluster)
                print("  🎯 cluster size \(cluster.count)")
            }
        }

        clusters.sort { $0.count > $1.count }
        if clusters.count > config.maxPatternsToDiscover {
            clusters = Array(clusters.prefix(config.maxPatternsToDiscover))
            print("  📏 Limited to top \(config.maxPatternsToDiscover) clusters")
        }
        return clusters
    }

    // MARK: - Pattern Creation

    private func createPatternFromCluster(_ cluster: [AudioSegment],
                                          allTrainingExamples: [TrainingExample]) -> CompositionalPattern? {
        guard cluster.count >= config.minPatternFrequency else { return nil }
        print("🏗️ Creating pattern from cluster of \(cluster.count) segments")

        let clusterEmbeddings = cluster.map { $0.embeddings }
        guard let rep = averageEmbedding(from: clusterEmbeddings) else {
            print("❌ Failed to compute representative embedding")
            return nil
        }

        // role-aware reduced labels, fallback to full meanings
        let meanings = deriveCanonicalMeaning(for: cluster, from: allTrainingExamples)
        let associated = meanings.isEmpty ? findAssociatedMeanings(for: cluster, in: allTrainingExamples) : meanings

        let avgConf = cluster.map(\.confidence).reduce(0, +) / Float(cluster.count)
        let posVar = calculatePositionVariance(for: cluster)
        let consistency = max(0.1, 1.0 - posVar)
        let adjusted = min(1.0, avgConf * consistency * Float(cluster.count) * 0.1)

        guard adjusted >= config.minPatternConfidence else {
            print("  ⚠️ Low confidence \(adjusted) < \(config.minPatternConfidence)")
            return nil
        }

        let pattern = CompositionalPattern(
            representativeEmbedding: rep,
            segments: cluster,
            associatedMeanings: associated,
            confidence: adjusted
        )

        print("  ✅ pattern: freq=\(pattern.frequency) conf=\(pattern.confidence) meanings=\(associated)")
        return pattern
    }

    private func averageEmbedding(from list: [[Float]]) -> [Float]? {
        guard let first = list.first, !first.isEmpty else { return nil }
        var out = Array(repeating: Float(0), count: first.count)
        for v in list {
            guard v.count == first.count else { return nil }
            for i in 0..<v.count { out[i] += v[i] }
        }
        let n = Float(list.count)
        for i in 0..<out.count { out[i] /= n }
        return out
    }

    private func findAssociatedMeanings(for cluster: [AudioSegment],
                                        in trainingExamples: [TrainingExample]) -> [String] {
        let byId = Dictionary(uniqueKeysWithValues: trainingExamples.map { ($0.id, $0.typicalExplanation) })
        let texts = cluster.compactMap { byId[$0.parentExampleId] }
        return Array(Set(texts))
    }

    // role-aware reduction: early cluster -> common prefix, late -> suffix
    private func deriveCanonicalMeaning(for cluster: [AudioSegment],
                                        from trainingExamples: [TrainingExample]) -> [String] {
        let exById = Dictionary(uniqueKeysWithValues: trainingExamples.map { ($0.id, $0) })
        let texts = cluster.compactMap { exById[$0.parentExampleId]?.typicalExplanation }
        guard !texts.isEmpty else { return [] }

        let tokensList = texts.map { $0.lowercased().split(separator: " ").map(String.init) }

        let avgStart = cluster.map(\.startTime).reduce(0, +) / Double(cluster.count)
        let maxEnd = cluster.map(\.endTime).max() ?? max(0.001, avgStart)
        let relPos = avgStart / maxEnd

        func lcp(_ lists: [[String]]) -> [String] {
            guard let f = lists.first else { return [] }
            var out: [String] = []
            for i in 0..<f.count {
                let t = f[i]
                if lists.allSatisfy({ $0.count > i && $0[i] == t }) { out.append(t) } else { break }
            }
            return out
        }
        func lcs(_ lists: [[String]]) -> [String] {
            let rev = lists.map { Array($0.reversed()) }
            return Array(lcp(rev).reversed())
        }

        let pref = lcp(tokensList)
        let suf = lcs(tokensList)

        let chosen: [String]
        if relPos < 0.35 { chosen = pref }
        else if relPos > 0.65 { chosen = suf }
        else { chosen = pref.count <= suf.count ? pref : suf }

        if chosen.isEmpty {
            let counts = tokensList.flatMap { $0 }.reduce(into: [String:Int]()) { $0[$1, default: 0] += 1 }
            if let top = counts.max(by: { $0.value < $1.value })?.key { return [top] }
        }
        let result = chosen.joined(separator: " ")
        return result.isEmpty ? [] : [result]
    }

    private func calculatePositionVariance(for cluster: [AudioSegment]) -> Float {
        let xs = cluster.map { Float($0.startTime) }
        let mean = xs.reduce(0, +) / Float(xs.count)
        let varSum = xs.reduce(0) { $0 + pow($1 - mean, 2) }
        return sqrt(varSum / Float(xs.count))
    }
}

// MARK: - Analysis Utilities

extension PatternDiscoveryEngine {
    func analyzePatternQuality(_ patterns: PatternCollection) -> PatternQualityReport {
        let total = patterns.patterns.count
        let sig = patterns.significantPatterns.count
        let avgFreq = total == 0 ? 0 : patterns.patterns.map(\.frequency).reduce(0, +) / total
        let avgConf: Float = total == 0 ? 0 : patterns.patterns.map(\.confidence).reduce(0, +) / Float(total)
        let coverage = total > 0 ? Float(sig) / Float(total) : 0

        return PatternQualityReport(
            totalPatterns: total,
            significantPatterns: sig,
            averageFrequency: avgFreq,
            averageConfidence: avgConf,
            coverage: coverage,
            recommendations: generateRecommendations(patterns)
        )
    }

    private func generateRecommendations(_ patterns: PatternCollection) -> [String] {
        var recs: [String] = []
        if patterns.significantPatterns.count < 3 {
            recs.append("Add more diverse training examples to discover additional patterns")
        }
        let lowConf = patterns.patterns.filter { $0.confidence < 0.5 }.count
        if lowConf > patterns.patterns.count / 2 {
            recs.append("Many patterns have low confidence, consider tuning similarity thresholds")
        }
        let lowFreq = patterns.patterns.filter { $0.frequency < 3 }.count
        if lowFreq > patterns.patterns.count / 3 {
            recs.append("Consider increasing minimum pattern frequency to reduce noise")
        }
        return recs
    }
}

// MARK: - Report

struct PatternQualityReport {
    let totalPatterns: Int
    let significantPatterns: Int
    let averageFrequency: Int
    let averageConfidence: Float
    let coverage: Float
    let recommendations: [String]
    var qualityScore: Float { (coverage * 0.4) + (averageConfidence * 0.6) }
}
