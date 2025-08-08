//
//  TrainingExample.swift
//  RelayTive
//
//  Training examples created by caregivers - these are the persistent, learnable mappings
//

import Foundation

struct TrainingExample: Identifiable, Codable {
    let id : UUID
    let atypicalAudio: Data              // Original atypical speech recording
    let typicalExplanation: String       // Caregiver's explanation in typical language
    let timestamp: Date                  // When this training example was created
    var isVerified: Bool                 // Whether caregiver has verified this mapping
    var audioEmbeddings: [Float]?        // HuBERT embeddings for the atypical audio (computed once, cached)
    
    init(atypicalAudio: Data, typicalExplanation: String, timestamp: Date = Date(), isVerified: Bool = false) {
        self.id = UUID()
        self.atypicalAudio = atypicalAudio
        self.typicalExplanation = typicalExplanation
        self.timestamp = timestamp
        self.isVerified = isVerified
        self.audioEmbeddings = nil
    }
    
    init(id: UUID, atypicalAudio: Data, typicalExplanation: String, timestamp: Date, isVerified: Bool = false, audioEmbeddings: [Float]? = nil) {
        self.id = id
        self.atypicalAudio = atypicalAudio
        self.typicalExplanation = typicalExplanation
        self.timestamp = timestamp
        self.isVerified = isVerified
        self.audioEmbeddings = audioEmbeddings
    }
}

// MARK: - Training Example Extensions
extension TrainingExample {
    var isRecent: Bool {
        let oneWeekAgo = Date().addingTimeInterval(-7 * 24 * 3600)
        return timestamp > oneWeekAgo
    }
    
    var hasEmbeddings: Bool {
        return audioEmbeddings != nil && !audioEmbeddings!.isEmpty
    }
    
    mutating func setEmbeddings(_ embeddings: [Float]) {
        self.audioEmbeddings = embeddings
    }
}

// MARK: - Translation Session (NOT stored permanently)
struct TranslationSession {
    let id = UUID()
    let audioData: Data
    let translationResult: String
    let confidence: Double
    let timestamp: Date
    let processingTime: TimeInterval
    
    // This represents a single translation attempt - NOT saved permanently
    // Only used for UI display during the translation session
}

// MARK: - Training Data Management
class TrainingDataManager {
    private let embeddingsSimilarityThreshold: Float = 0.70
    
    /// Find the best matching training example for given audio embeddings
    func findBestMatch(for embeddings: [Float], in trainingExamples: [TrainingExample]) -> (example: TrainingExample, similarity: Float)? {
        guard !embeddings.isEmpty else { return nil }
        
        var bestMatch: TrainingExample?
        var highestSimilarity: Float = 0.0
        
        for example in trainingExamples {
            guard let exampleEmbeddings = example.audioEmbeddings else { continue }
            
            let similarity = cosineSimilarity(embeddings, exampleEmbeddings)
            
            if similarity > highestSimilarity && similarity > embeddingsSimilarityThreshold {
                highestSimilarity = similarity
                bestMatch = example
            }
        }
        
        guard let match = bestMatch else { return nil }
        return (match, highestSimilarity)
    }
    
    /// Calculate cosine similarity between two embedding vectors
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count && !a.isEmpty else { return 0.0 }
        
        let dotProduct = zip(a, b).map(*).reduce(0, +)
        let magnitudeA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        guard magnitudeA > 0 && magnitudeB > 0 else { return 0.0 }
        
        return dotProduct / (magnitudeA * magnitudeB)
    }
}
