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
    @Published var compositionalPatterns: PatternCollection = PatternCollection() // Discovered patterns
    
    private let userDefaults = UserDefaults.standard
    private let trainingExamplesKey = "TrainingExamples"
    private let compositionalPatternsKey = "CompositionalPatterns"
    private let trainingDataManager = TrainingDataManager()
    
    // Compositional pipeline components
    private var segmentationEngine: AudioSegmentationEngine?
    private var patternDiscoveryEngine: PatternDiscoveryEngine?
    private var compositionalMatcher: CompositionalMatcher?
    private var isProcessingPatterns = false // Prevent concurrent pattern discovery
    
    init() {
        loadTrainingExamples()
        loadCompositionalPatterns()
    }
    
    // MARK: - Compositional Pipeline Setup
    
    func initializeCompositionalPipeline(with translationEngine: TranslationEngine) {
        print("🚀 Initializing compositional pipeline")
        
        segmentationEngine = AudioSegmentationEngine(translationEngine: translationEngine)
        
        if let segEngine = segmentationEngine {
            patternDiscoveryEngine = PatternDiscoveryEngine(segmentationEngine: segEngine)
            compositionalMatcher = CompositionalMatcher(segmentationEngine: segEngine)
        }
        
        print("✅ Compositional pipeline initialized")
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
    
    // MARK: - Translation Lookup (Compositional + Fallback)
    
    /// Enhanced translation lookup using compositional patterns with fallback to whole-utterance matching
    func findTranslationForAudio(_ audioData: Data, embeddings: [Float], using translationEngine: TranslationEngine) async -> (translation: String, confidence: Float)? {
        
        // Ensure compositional pipeline is initialized
        if segmentationEngine == nil {
            initializeCompositionalPipeline(with: translationEngine)
        }
        
        guard let matcher = compositionalMatcher else {
            print("⚠️ Compositional matcher not available, falling back to traditional matching")
            return findTranslationForEmbeddings(embeddings)
        }
        
        // Try compositional matching first
        print("🎯 Attempting compositional pattern matching")
        let matchResult = await matcher.matchAudio(audioData, 
                                                 against: compositionalPatterns, 
                                                 fallbackExamples: examplesWithEmbeddings)
        
        if matchResult.hasMatches {
            print("✅ Compositional match found: \(matchResult.reconstructedTranslation) (confidence: \(matchResult.overallConfidence))")
            return (matchResult.reconstructedTranslation, matchResult.overallConfidence)
        }
        
        // Fallback to traditional whole-utterance matching
        print("🔄 Compositional matching failed, using traditional approach")
        return findTranslationForEmbeddings(embeddings)
    }
    
    /// Legacy method for whole-utterance matching (kept for backwards compatibility)
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
    
    // MARK: - Compositional Pattern Management
    
    /// Perform full pattern discovery across all training examples
    func performCompositionalPatternDiscovery(using translationEngine: TranslationEngine) async {
        print("🔍 Starting compositional pattern discovery")
        
        // Prevent running if already processing
        if isProcessingPatterns {
            print("⚠️ Pattern discovery already in progress, skipping")
            return
        }
        
        isProcessingPatterns = true
        defer { isProcessingPatterns = false }
        
        // Ensure pipeline is initialized
        if patternDiscoveryEngine == nil {
            initializeCompositionalPipeline(with: translationEngine)
        }
        
        guard let discoveryEngine = patternDiscoveryEngine else {
            print("❌ Pattern discovery engine not available")
            return
        }
        
        // Only discover patterns if we have sufficient training data
        guard examplesWithEmbeddings.count >= 2 && examplesWithEmbeddings.count <= 20 else {
            if examplesWithEmbeddings.count < 2 {
                print("⚠️ Need at least 2 examples with embeddings for pattern discovery")
            } else {
                print("⚠️ Too many examples (\(examplesWithEmbeddings.count)), limiting pattern discovery")
            }
            return
        }
        
        // Run pattern discovery
        let discoveredPatterns = await discoveryEngine.discoverPatterns(from: examplesWithEmbeddings)
        
        // Update our pattern collection
        await MainActor.run {
            compositionalPatterns = discoveredPatterns
            saveCompositionalPatterns()
            
            print("✅ Pattern discovery complete: \(compositionalPatterns.significantPatterns.count) significant patterns found")
            
            // Log pattern quality analysis
            let qualityReport = discoveryEngine.analyzePatternQuality(compositionalPatterns)
            print("📊 Pattern Quality Report:")
            print("  - Total patterns: \(qualityReport.totalPatterns)")
            print("  - Significant patterns: \(qualityReport.significantPatterns)")  
            print("  - Average confidence: \(qualityReport.averageConfidence)")
            print("  - Quality score: \(qualityReport.qualityScore)")
            
            for recommendation in qualityReport.recommendations {
                print("  💡 \(recommendation)")
            }
        }
    }
    
    /// Update patterns incrementally when new training examples are added
    func updateCompositionalPatterns(with newExamples: [TrainingExample], using translationEngine: TranslationEngine) async {
        guard let discoveryEngine = patternDiscoveryEngine,
              !compositionalPatterns.patterns.isEmpty else {
            // No existing patterns, trigger full discovery
            await performCompositionalPatternDiscovery(using: translationEngine)
            return
        }
        
        print("🔄 Updating compositional patterns with \(newExamples.count) new examples")
        
        let updatedPatterns = await discoveryEngine.updatePatterns(compositionalPatterns, with: newExamples)
        
        await MainActor.run {
            compositionalPatterns = updatedPatterns
            saveCompositionalPatterns()
            print("✅ Patterns updated: \(compositionalPatterns.significantPatterns.count) significant patterns")
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
    
    private func saveCompositionalPatterns() {
        do {
            let data = try JSONEncoder().encode(compositionalPatterns)
            userDefaults.set(data, forKey: compositionalPatternsKey)
            print("Saved \(compositionalPatterns.patterns.count) compositional patterns to disk")
        } catch {
            print("Failed to save compositional patterns: \(error)")
        }
    }
    
    private func loadCompositionalPatterns() {
        guard let data = userDefaults.data(forKey: compositionalPatternsKey) else {
            compositionalPatterns = PatternCollection()
            print("No existing compositional patterns found - starting fresh")
            return
        }
        
        do {
            compositionalPatterns = try JSONDecoder().decode(PatternCollection.self, from: data)
            print("Loaded \(compositionalPatterns.patterns.count) compositional patterns from disk")
        } catch {
            print("Failed to load compositional patterns: \(error)")
            compositionalPatterns = PatternCollection()
        }
    }
    
    // MARK: - Development Helpers
    
    func clearAllTrainingData() {
        trainingExamples = []
        compositionalPatterns = PatternCollection()
        userDefaults.removeObject(forKey: trainingExamplesKey)
        userDefaults.removeObject(forKey: compositionalPatternsKey)
        print("All training data and compositional patterns cleared")
    }
    
    // NOTE: No sample data loading - app must start completely empty
    
    // MARK: - Legacy Compatibility (for Views that still reference old model)
    
    var allUtterances: [Utterance] {
        // Convert TrainingExamples to Utterances for backward compatibility
        // IMPORTANT: Use the TrainingExample's ID to maintain consistency
        return trainingExamples.map { example in
            Utterance(
                id: example.id,
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
        // Find corresponding TrainingExample by ID and update both translation and verification
        if let index = trainingExamples.firstIndex(where: { $0.id == updatedUtterance.id }) {
            // Create a new TrainingExample with updated values
            let originalExample = trainingExamples[index]
            let updatedExample = TrainingExample(
                id: originalExample.id,
                atypicalAudio: originalExample.atypicalAudio,
                typicalExplanation: updatedUtterance.translation, // Update translation
                timestamp: originalExample.timestamp,
                isVerified: updatedUtterance.isVerified, // Update verification status
                audioEmbeddings: originalExample.audioEmbeddings // Preserve embeddings
            )
            
            trainingExamples[index] = updatedExample
            saveTrainingExamples()
            print("Training example updated - translation: '\(updatedExample.typicalExplanation)', verified: \(updatedExample.isVerified)")
        } else {
            print("Warning: Could not find TrainingExample with ID \(updatedUtterance.id) to update")
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