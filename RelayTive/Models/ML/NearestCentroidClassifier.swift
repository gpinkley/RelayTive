//
//  NearestCentroidClassifier.swift
//  RelayTive
//
//  Nearest centroid classifier with phonetic similarity fusion and online learning
//

import Foundation
import Accelerate

/// Nearest centroid classifier optimized for speech recognition with phonetic similarity
class NearestCentroidClassifier {
    
    // MARK: - Configuration
    private let temperature: Float = 10.0
    private let embeddingWeight: Float = 0.6
    private let phoneticWeight: Float = 0.4
    private let confidenceThreshold: Float = 0.70
    private let marginThreshold: Float = 0.10
    private let maxPrototypesPerMeaning = 10
    
    // MARK: - State
    private var meaningCentroids: [String: [Float]] = [:]
    private var phoneticPrototypes: [String: [String]] = [:]
    private var updateCounts: [String: Int] = [:]
    
    private let queue = DispatchQueue(label: "com.relayTive.classifier", qos: .utility)
    
    init() {
        print("🎯 NearestCentroidClassifier initialized")
    }
    
    /// Classify an embedding with optional phonetic string
    func classify(embedding: [Float], phoneticString: String? = nil) -> ClassificationResult {
        return queue.sync {
            guard !meaningCentroids.isEmpty else {
                return ClassificationResult(
                    topMeaning: nil,
                    confidence: 0.0,
                    margin: 0.0,
                    alternatives: [],
                    needsConfirmation: true,
                    embeddingConfidence: 0.0,
                    phoneticConfidence: 0.0
                )
            }
            
            let normalizedEmbedding = l2Normalize(embedding)
            let embeddingSimilarities = calculateEmbeddingSimilarities(normalizedEmbedding)
            
            var phoneticSimilarities: [String: Float] = [:]
            if let phoneticStr = phoneticString {
                phoneticSimilarities = calculatePhoneticSimilarities(phoneticStr)
            }
            
            let fusedSimilarities = fuseSimilarities(
                embeddingSimilarities: embeddingSimilarities,
                phoneticSimilarities: phoneticSimilarities
            )
            
            let probabilities = temperatureSoftmax(fusedSimilarities, temperature: temperature)
            let sortedResults = probabilities.sorted { $0.value > $1.value }
            
            guard let topResult = sortedResults.first else {
                return ClassificationResult(
                    topMeaning: nil,
                    confidence: 0.0,
                    margin: 0.0,
                    alternatives: [],
                    needsConfirmation: true,
                    embeddingConfidence: 0.0,
                    phoneticConfidence: 0.0
                )
            }
            
            let topMeaning = topResult.key
            let topConfidence = topResult.value
            let margin = sortedResults.count > 1 ? topConfidence - sortedResults[1].value : topConfidence
            let needsConfirmation = topConfidence < confidenceThreshold || margin < marginThreshold
            
            let alternatives = Array(sortedResults.dropFirst().prefix(2)).map {
                Alternative(meaning: $0.key, confidence: $0.value)
            }
            
            return ClassificationResult(
                topMeaning: topMeaning,
                confidence: topConfidence,
                margin: margin,
                alternatives: alternatives,
                needsConfirmation: needsConfirmation,
                embeddingConfidence: embeddingSimilarities[topMeaning] ?? 0.0,
                phoneticConfidence: phoneticSimilarities[topMeaning] ?? 0.0
            )
        }
    }
    
    /// Update classifier with caregiver feedback
    func updateWithExample(meaning: String, embedding: [Float], phoneticString: String?) {
        queue.sync {
            let normalizedEmbedding = l2Normalize(embedding)
            updateEmbeddingCentroid(meaning: meaning, embedding: normalizedEmbedding)
            
            if let phoneticStr = phoneticString, !phoneticStr.isEmpty {
                updatePhoneticPrototypes(meaning: meaning, phoneticString: phoneticStr)
            }
            
            updateCounts[meaning, default: 0] += 1
        }
        
        print("🎯 Classifier updated for meaning: '\(meaning)'")
    }
    
    func getKnownMeanings() -> [String] {
        return queue.sync { Array(meaningCentroids.keys).sorted() }
    }
    
    func reset() {
        queue.sync {
            meaningCentroids.removeAll()
            phoneticPrototypes.removeAll()
            updateCounts.removeAll()
        }
        print("🎯 NearestCentroidClassifier reset")
    }
    
    // MARK: - Private Methods
    
    private func calculateEmbeddingSimilarities(_ embedding: [Float]) -> [String: Float] {
        var similarities: [String: Float] = [:]
        for (meaning, centroid) in meaningCentroids {
            similarities[meaning] = dotProduct(embedding, centroid)
        }
        return similarities
    }
    
    private func calculatePhoneticSimilarities(_ phoneticString: String) -> [String: Float] {
        var similarities: [String: Float] = [:]
        for (meaning, prototypes) in phoneticPrototypes {
            let maxSimilarity = prototypes.map { prototype in
                calculatePhoneticSimilarity(phoneticString, prototype)
            }.max() ?? 0.0
            similarities[meaning] = maxSimilarity
        }
        return similarities
    }
    
    private func calculatePhoneticSimilarity(_ string1: String, _ string2: String) -> Float {
        let distance = levenshteinDistance(string1, string2)
        let maxLength = max(string1.count, string2.count)
        guard maxLength > 0 else { return 1.0 }
        return max(0.0, 1.0 - Float(distance) / Float(maxLength))
    }
    
    private func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let a1 = Array(s1), a2 = Array(s2)
        let m = a1.count, n = a2.count
        if m == 0 { return n }
        if n == 0 { return m }
        
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        
        for i in 1...m {
            for j in 1...n {
                let cost = a1[i-1] == a2[j-1] ? 0 : 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
            }
        }
        return dp[m][n]
    }
    
    private func fuseSimilarities(embeddingSimilarities: [String: Float], phoneticSimilarities: [String: Float]) -> [String: Float] {
        var fusedSimilarities: [String: Float] = [:]
        for meaning in meaningCentroids.keys {
            let embeddingSim = embeddingSimilarities[meaning] ?? 0.0
            let phoneticSim = phoneticSimilarities[meaning] ?? 0.0
            fusedSimilarities[meaning] = embeddingWeight * embeddingSim + phoneticWeight * phoneticSim
        }
        return fusedSimilarities
    }
    
    private func temperatureSoftmax(_ similarities: [String: Float], temperature: Float) -> [String: Float] {
        guard !similarities.isEmpty else { return [:] }
        let maxSim = similarities.values.max() ?? 0.0
        
        var exponentials: [String: Float] = [:]
        var sum: Float = 0.0
        
        for (meaning, similarity) in similarities {
            let exp = expf((similarity - maxSim) / temperature)
            exponentials[meaning] = exp
            sum += exp
        }
        
        var probabilities: [String: Float] = [:]
        for (meaning, exp) in exponentials {
            probabilities[meaning] = sum > 0 ? exp / sum : 0.0
        }
        return probabilities
    }
    
    private func updateEmbeddingCentroid(meaning: String, embedding: [Float]) {
        let updateCount = updateCounts[meaning] ?? 0
        let learningRate = max(0.05, 0.3 * pow(0.95, Float(updateCount) / 10.0))
        
        if var existingCentroid = meaningCentroids[meaning] {
            let oneMinusLR = 1.0 - learningRate
            for i in 0..<min(existingCentroid.count, embedding.count) {
                existingCentroid[i] = oneMinusLR * existingCentroid[i] + learningRate * embedding[i]
            }
            meaningCentroids[meaning] = l2Normalize(existingCentroid)
        } else {
            meaningCentroids[meaning] = embedding
        }
    }
    
    private func updatePhoneticPrototypes(meaning: String, phoneticString: String) {
        var prototypes = phoneticPrototypes[meaning] ?? []
        prototypes.append(phoneticString)
        if prototypes.count > maxPrototypesPerMeaning {
            prototypes = Array(prototypes.suffix(maxPrototypesPerMeaning))
        }
        phoneticPrototypes[meaning] = prototypes
    }
    
    private func l2Normalize(_ vector: [Float]) -> [Float] {
        var result = vector
        var norm: Float = 0.0
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        if norm > 1e-10 {
            vDSP_vsdiv(vector, 1, &norm, &result, 1, vDSP_Length(vector.count))
        }
        return result
    }
    
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0.0 }
        var result: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
}

// MARK: - Support Structures

struct ClassificationResult {
    let topMeaning: String?
    let confidence: Float
    let margin: Float
    let alternatives: [Alternative]
    let needsConfirmation: Bool
    let embeddingConfidence: Float
    let phoneticConfidence: Float
    
    var isConfident: Bool {
        return !needsConfirmation && confidence >= 0.70
    }
}

struct Alternative {
    let meaning: String
    let confidence: Float
}
