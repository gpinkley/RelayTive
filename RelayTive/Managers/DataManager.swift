//
//  DataManager.swift
//  RelayTive
//
//  Manages training examples (persistent) separate from ad-hoc translations (temporary)
//

import Foundation
import SwiftUI

@MainActor
class DataManager: ObservableObject {
    @Published var trainingExamples: [TrainingExample] = []  // Only examples from Training tab
    
    private let userDefaults = UserDefaults.standard
    private let trainingExamplesKey = "TrainingExamples"
    private let trainingDataManager = TrainingDataManager()
    
    init() {
        loadTrainingExamples()
    }
    
    // MARK: - Training Example Management (Persistent Data)
    
    func addTrainingExample(_ example: TrainingExample) {
        trainingExamples.insert(example, at: 0) // Most recent first
        saveTrainingExamples()
        print("Training example added: \(example.typicalExplanation)")
    }
    
    func updateTrainingExample(_ updatedExample: TrainingExample) {
        if let index = trainingExamples.firstIndex(where: { $0.id == updatedExample.id }) {
            trainingExamples[index] = updatedExample
            saveTrainingExamples()
            print("Training example updated: \(updatedExample.typicalExplanation)")
        }
    }
    
    func removeTrainingExample(_ example: TrainingExample) {
        trainingExamples.removeAll { $0.id == example.id }
        saveTrainingExamples()
        print("Training example removed")
    }
    
    // MARK: - Translation Lookup (Uses Training Data Only)
    
    func findTranslationForEmbeddings(_ embeddings: [Float]) -> (translation: String, confidence: Float)? {
        guard let match = trainingDataManager.findBestMatch(for: embeddings, in: trainingExamples) else {
            print("No matching training example found for embeddings")
            return nil
        }
        
        print("Found matching training example: \(match.example.typicalExplanation) (similarity: \(match.similarity))")
        return (match.example.typicalExplanation, match.similarity)
    }
    
    func addEmbeddingsToExample(id: UUID, embeddings: [Float]) {
        if let index = trainingExamples.firstIndex(where: { $0.id == id }) {
            trainingExamples[index].setEmbeddings(embeddings)
            saveTrainingExamples()
            print("Embeddings added to training example: \(trainingExamples[index].typicalExplanation)")
        }
    }
    
    // MARK: - Training Data Statistics
    
    var unverifiedExamples: [TrainingExample] {
        return trainingExamples.filter { !$0.isVerified }
    }
    
    var verifiedExamples: [TrainingExample] {
        return trainingExamples.filter(\.isVerified)
    }
    
    var examplesWithEmbeddings: [TrainingExample] {
        return trainingExamples.filter(\.hasEmbeddings)
    }
    
    // MARK: - Training Statistics
    
    var totalTrainingExamples: Int {
        return trainingExamples.count
    }
    
    var verificationRate: Double {
        guard totalTrainingExamples > 0 else { return 0.0 }
        let verifiedCount = verifiedExamples.count
        return Double(verifiedCount) / Double(totalTrainingExamples)
    }
    
    var embeddingsCompletionRate: Double {
        guard totalTrainingExamples > 0 else { return 0.0 }
        let embeddingsCount = examplesWithEmbeddings.count
        return Double(embeddingsCount) / Double(totalTrainingExamples)
    }
    
    // MARK: - Persistence
    
    private func saveTrainingExamples() {
        do {
            let data = try JSONEncoder().encode(trainingExamples)
            userDefaults.set(data, forKey: trainingExamplesKey)
            print("Saved \(trainingExamples.count) training examples to disk")
        } catch {
            print("Failed to save training examples: \(error)")
        }
    }
    
    private func loadTrainingExamples() {
        guard let data = userDefaults.data(forKey: trainingExamplesKey) else {
            // Start completely empty - no predefined examples
            trainingExamples = []
            print("No existing training examples found - starting fresh")
            return
        }
        
        do {
            trainingExamples = try JSONDecoder().decode([TrainingExample].self, from: data)
            print("Loaded \(trainingExamples.count) training examples from disk")
        } catch {
            print("Failed to load training examples: \(error)")
            trainingExamples = []
        }
    }
    
    // MARK: - Development Helpers
    
    func clearAllTrainingData() {
        trainingExamples = []
        userDefaults.removeObject(forKey: trainingExamplesKey)
        print("All training data cleared")
    }
    
    // NOTE: No sample data loading - app must start completely empty
    
    // MARK: - Legacy Compatibility (for Views that still reference old model)
    
    var allUtterances: [Utterance] {
        // Convert TrainingExamples to Utterances for backward compatibility
        return trainingExamples.map { example in
            Utterance(
                originalAudio: example.atypicalAudio,
                translation: example.typicalExplanation,
                timestamp: example.timestamp,
                isVerified: example.isVerified
            )
        }
    }
    
    func addUtterance(_ utterance: Utterance) {
        // Convert Utterance to TrainingExample
        let example = TrainingExample(
            atypicalAudio: utterance.originalAudio,
            typicalExplanation: utterance.translation,
            timestamp: utterance.timestamp,
            isVerified: utterance.isVerified
        )
        addTrainingExample(example)
    }
    
    func updateUtterance(_ updatedUtterance: Utterance) {
        // Find corresponding TrainingExample and update
        if let index = trainingExamples.firstIndex(where: { 
            $0.id == updatedUtterance.id || 
            ($0.timestamp == updatedUtterance.timestamp && $0.typicalExplanation == updatedUtterance.translation) 
        }) {
            var updatedExample = trainingExamples[index]
            updatedExample.isVerified = updatedUtterance.isVerified
            updateTrainingExample(updatedExample)
        }
    }
    
    func removeUtterance(_ utterance: Utterance) {
        // Find corresponding TrainingExample and remove
        if let example = trainingExamples.first(where: { 
            $0.timestamp == utterance.timestamp && $0.typicalExplanation == utterance.translation 
        }) {
            removeTrainingExample(example)
        }
    }
}