//
//  TrainingExample.swift
//  RelayTive
//
//  Training examples created by caregivers - these are the persistent, learnable mappings
//

import Foundation

struct TrainingExample: Identifiable, Codable {
    let id : UUID
    let atypicalAudio: Data              // Original atypical speech recording (legacy)
    let typicalExplanation: String       // Caregiver's explanation in typical language
    let timestamp: Date                  // When this training example was created
    var isVerified: Bool                 // Whether caregiver has verified this mapping
    var audioEmbeddings: [Float]?        // HuBERT embeddings for the atypical audio (computed once, cached)
    var audioFileURL: URL?               // Persistent audio file URL for full recordings
    
    // Phonetic data
    var phoneticForm: PhoneticForm?      // Phonetic transcription and metadata
    
    init(atypicalAudio: Data, typicalExplanation: String, timestamp: Date = Date(), isVerified: Bool = false) {
        self.id = UUID()
        self.atypicalAudio = atypicalAudio
        self.typicalExplanation = typicalExplanation
        self.timestamp = timestamp
        self.isVerified = isVerified
        self.audioEmbeddings = nil
        self.audioFileURL = nil
        self.phoneticForm = nil
    }
    
    init(id: UUID, atypicalAudio: Data, typicalExplanation: String, timestamp: Date, isVerified: Bool = false, audioEmbeddings: [Float]? = nil, audioFileURL: URL? = nil, phoneticForm: PhoneticForm? = nil) {
        self.id = id
        self.atypicalAudio = atypicalAudio
        self.typicalExplanation = typicalExplanation
        self.timestamp = timestamp
        self.isVerified = isVerified
        self.audioEmbeddings = audioEmbeddings
        self.audioFileURL = audioFileURL
        self.phoneticForm = phoneticForm
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
    
    var hasPhoneticForm: Bool {
        return phoneticForm != nil
    }
    
    mutating func setEmbeddings(_ embeddings: [Float]) {
        self.audioEmbeddings = embeddings
    }
    
    mutating func setPhoneticForm(_ form: PhoneticForm) {
        self.phoneticForm = form
    }
    
    /// Migrate audio data to file URL if needed
    mutating func migrateAudioToFile() {
        guard audioFileURL == nil, !atypicalAudio.isEmpty else { return }
        
        do {
            let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let trainingAudioDir = documentsDir.appendingPathComponent("TrainingAudio")
            
            // Ensure directory exists
            try FileManager.default.createDirectory(at: trainingAudioDir, withIntermediateDirectories: true, attributes: nil)
            
            let audioFile = trainingAudioDir.appendingPathComponent("\(id.uuidString).wav")
            try atypicalAudio.write(to: audioFile)
            
            audioFileURL = audioFile
            print("📁 Migrated audio data to file: \(audioFile.lastPathComponent)")
        } catch {
            print("❌ Failed to migrate audio to file: \(error)")
        }
    }
    
    /// Get audio data, preferring file URL over embedded data
    var audioData: Data? {
        if let fileURL = audioFileURL {
            return try? Data(contentsOf: fileURL)
        }
        return atypicalAudio.isEmpty ? nil : atypicalAudio
    }
    
    /// Get audio file URL, creating it if necessary
    mutating func getOrCreateAudioFileURL() -> URL? {
        if audioFileURL == nil {
            migrateAudioToFile()
        }
        return audioFileURL
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
// MARK: - Phonetic Form
struct PhoneticForm: Codable {
    let unitIDs: [Int]                  // Discrete phonetic unit IDs
    let unitString: String              // "U12 U7 U3" format
    let canonicalUnitsString: String    // Canonical/cleaned version
    let readableSpelling: String?       // Optional symbol mapping
    let confidence: Float               // Transcription confidence
    let timestamp: Date                 // When transcribed
    
    init(unitIDs: [Int], unitString: String, readableSpelling: String? = nil, confidence: Float = 1.0) {
        self.unitIDs = unitIDs
        self.unitString = unitString
        self.canonicalUnitsString = unitString // Could be different after processing
        self.readableSpelling = readableSpelling
        self.confidence = confidence
        self.timestamp = Date()
    }
}

// MARK: - User Profile
struct UserProfile: Codable {
    var phoneticData: PhoneticProfileData?
    var preferences: UserPreferences
    var lastUpdated: Date
    
    init() {
        self.phoneticData = nil
        self.preferences = UserPreferences()
        self.lastUpdated = Date()
    }
}

struct UserPreferences: Codable {
    var confidenceThreshold: Float = 0.70
    var enablePhoneticTranscription: Bool = true
    var debugMode: Bool = false
    
    init() {}
}

struct PhoneticProfileData: Codable {
    let codebook: [[Float]]             // K-means centroids
    let unitToSymbol: [String: String]  // Unit ID -> symbol mapping (string keys for JSON)
    let meaningCentroids: [String: [Float]]     // Meaning -> centroid mapping
    let phoneticPrototypes: [String: [String]]  // Meaning -> phonetic prototypes
    let lastUpdated: Date
    
    init(codebook: [[Float]], unitToSymbol: [Int: String], meaningCentroids: [String: [Float]], phoneticPrototypes: [String: [String]]) {
        self.codebook = codebook
        self.unitToSymbol = unitToSymbol.reduce(into: [:]) { result, pair in
            result[String(pair.key)] = pair.value
        }
        self.meaningCentroids = meaningCentroids
        self.phoneticPrototypes = phoneticPrototypes
        self.lastUpdated = Date()
    }
    
    func getUnitToSymbolMap() -> [Int: String] {
        return unitToSymbol.reduce(into: [:]) { result, pair in
            if let key = Int(pair.key) {
                result[key] = pair.value
            }
        }
    }
}

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
