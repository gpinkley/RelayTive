//
//  DataManager.swift
//  RelayTive
//
//  Manages training examples (persistent) separate from ad-hoc translations (temporary)
//

import Foundation
import SwiftUI
import AVFoundation

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
    private var patternDiscoveryEnabled = true // Can be disabled for memory efficiency
    private var pendingDiscoveryTask: Task<Void, Never>? // Debounce task
    private var newExampleCount = 0 // Track new examples for triggering discovery
    
    // Phonetic pipeline components
    private var phoneticEngine: PhoneticTranscriptionEngine?
    private var phoneticClassifier: NearestCentroidClassifier?
    
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
    
    func initializePhoneticPipeline(with translationEngine: TranslationEngine) {
        print("🚀 Initializing phonetic pipeline")
        
        phoneticEngine = PhoneticTranscriptionEngine(translationEngine: translationEngine, embeddingDim: 768, k: 160)
        phoneticClassifier = NearestCentroidClassifier()
        
        print("✅ Phonetic pipeline initialized")
    }
    
    // MARK: - Training Example Management (Persistent Data)
    
    func addTrainingExample(_ example: TrainingExample) {
        trainingExamples.insert(example, at: 0) // Most recent first
        
        // Keep only recent examples in memory to prevent memory bloat
        if trainingExamples.count > 15 {
            let removedCount = trainingExamples.count - 15
            trainingExamples = Array(trainingExamples.prefix(15))
            print("🧹 Memory management: kept recent 15 examples, removed \(removedCount) older ones")
        }
        
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
    
    /// Enhanced translation lookup using phonetic pipeline with compositional and traditional fallbacks
    func findTranslationForAudio(_ audioData: Data, embeddings: [Float], using translationEngine: TranslationEngine) async -> (translation: String, confidence: Float)? {
        
        // Ensure pipelines are initialized
        if segmentationEngine == nil {
            initializeCompositionalPipeline(with: translationEngine)
        }
        if phoneticEngine == nil {
            initializePhoneticPipeline(with: translationEngine)
        }
        
        // Try phonetic classification first (most accurate)
        if let phoneticResult = await tryPhoneticClassification(audioData, using: translationEngine) {
            print("✅ Phonetic classification match: \(phoneticResult.translation) (confidence: \(phoneticResult.confidence))")
            return phoneticResult
        }
        
        // Fallback to compositional matching
        if let matcher = compositionalMatcher {
            print("🎯 Attempting compositional pattern matching")
            let matchResult = await matcher.matchAudio(audioData, 
                                                     against: compositionalPatterns, 
                                                     fallbackExamples: examplesWithEmbeddings)
            
            if matchResult.hasMatches {
                print("✅ Compositional match found: \(matchResult.reconstructedTranslation) (confidence: \(matchResult.overallConfidence))")
                return (matchResult.reconstructedTranslation, matchResult.overallConfidence)
            }
        }
        
        // Final fallback to traditional whole-utterance matching
        print("🔄 Using traditional embedding matching")
        return findTranslationForEmbeddings(embeddings)
    }
    
    /// Try phonetic classification approach
    private func tryPhoneticClassification(_ audioData: Data, using translationEngine: TranslationEngine) async -> (translation: String, confidence: Float)? {
        guard let engine = phoneticEngine,
              let classifier = phoneticClassifier else {
            return nil
        }
        
        // Convert audioData to buffer
        guard let buffer = createAudioBufferFromData(audioData) else {
            return nil
        }
        
        // Get phonetic transcription
        let tx = await engine.transcribe(buffer: buffer)
        
        guard !tx.unitString.isEmpty else {
            return nil
        }
        
        // Classify using embedding (extract from first chunk if available)
        let embedding = await extractRepresentativeEmbedding(from: buffer, using: translationEngine)
        
        guard let embeddingVector = embedding else {
            return nil
        }
        
        // Classify with phonetic string fusion
        let classification = classifier.classify(
            embedding: embeddingVector,
            phoneticString: tx.unitString
        )
        
        if let meaning = classification.topMeaning, !classification.needsConfirmation {
            return (meaning, classification.confidence)
        }
        
        return nil
    }
    
    /// Handle caregiver confirmation for phonetic pipeline
    func confirmPhoneticTranslation(audioData: Data, meaning: String, using translationEngine: TranslationEngine) async {
        guard let engine = phoneticEngine,
              let classifier = phoneticClassifier else {
            return
        }
        
        guard let buffer = createAudioBufferFromData(audioData) else {
            return
        }
        
        // Get phonetic transcription
        let tx = await engine.transcribe(buffer: buffer)
        
        // Extract embedding
        if let embedding = await extractRepresentativeEmbedding(from: buffer, using: translationEngine) {
            // Update classifier with confirmed example
            classifier.updateWithExample(
                meaning: meaning,
                embedding: embedding,
                phoneticString: tx.unitString
            )
            
            print("🎯 Phonetic classifier updated with confirmed example: '\(meaning)'")
        }
    }
    
    private func createAudioBufferFromData(_ audioData: Data) -> AVAudioPCMBuffer? {
        // Simplified buffer creation - assumes 16kHz mono PCM
        let frameCount = audioData.count / 2
        guard let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            return nil
        }
        
        buffer.frameLength = AVAudioFrameCount(frameCount)
        
        audioData.withUnsafeBytes { bytes in
            let int16Ptr = bytes.bindMemory(to: Int16.self)
            let floatPtr = buffer.floatChannelData![0]
            
            for i in 0..<frameCount {
                floatPtr[i] = Float(int16Ptr[i]) / 32768.0
            }
        }
        
        return buffer
    }
    
    private func extractRepresentativeEmbedding(from buffer: AVAudioPCMBuffer, using translationEngine: TranslationEngine) async -> [Float]? {
        return await translationEngine.extractFrameEmbedding(from: buffer)
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
    
    /// Trigger debounced pattern discovery
    func triggerPatternDiscovery(using translationEngine: TranslationEngine) {
        // Cancel any pending discovery task
        pendingDiscoveryTask?.cancel()
        
        // Only trigger every 2 new examples to reduce churn
        newExampleCount += 1
        guard newExampleCount >= 2 else {
            print("🔍 Pattern discovery delayed until \(2 - newExampleCount) more examples added")
            return
        }
        
        // Debounce with 1 second delay
        pendingDiscoveryTask = Task {
            try? await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            
            guard !Task.isCancelled else { return }
            
            await performCompositionalPatternDiscovery(using: translationEngine)
            await MainActor.run {
                newExampleCount = 0 // Reset counter after discovery
            }
        }
    }
    
    /// Perform full pattern discovery across all training examples
    func performCompositionalPatternDiscovery(using translationEngine: TranslationEngine) async {
        print("🔍 Starting compositional pattern discovery")
        
        // Skip if pattern discovery is disabled for memory efficiency
        if !patternDiscoveryEnabled {
            print("⚠️ Pattern discovery disabled for memory efficiency")
            return
        }
        
        // Re-enabled pattern discovery with memory and performance fixes
        
        // Prevent running if already processing
        guard !isProcessingPatterns else {
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
        
        // Only discover patterns if we have sufficient training data - limit to 8 examples for memory efficiency
        guard examplesWithEmbeddings.count >= 2 && examplesWithEmbeddings.count <= 8 else {
            if examplesWithEmbeddings.count < 2 {
                print("⚠️ Need at least 2 examples with embeddings for pattern discovery")
            } else {
                print("⚠️ Too many examples (\(examplesWithEmbeddings.count)), limiting pattern discovery for memory efficiency")
            }
            return
        }
        
        // Run pattern discovery on limited dataset for memory efficiency
        let limitedExamples = Array(examplesWithEmbeddings.suffix(8)) // Use only most recent 8
        let discoveredPatterns = await discoveryEngine.discoverPatterns(from: limitedExamples)
        
        // Update our pattern collection
        await MainActor.run {
            compositionalPatterns = discoveredPatterns
            
            // Aggressively prune to maintain memory efficiency
            compositionalPatterns.aggressivePrune()
            
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
    
    // MARK: - Memory Management
    
    /// Disable pattern discovery to save memory
    func disablePatternDiscovery() {
        patternDiscoveryEnabled = false
        print("🧹 Pattern discovery disabled for memory efficiency")
    }
    
    /// Enable pattern discovery 
    func enablePatternDiscovery() {
        patternDiscoveryEnabled = true
        print("🚀 Pattern discovery enabled")
    }
    
    /// Force cleanup of patterns to free memory
    func cleanupPatterns() {
        compositionalPatterns.aggressivePrune()
        saveCompositionalPatterns()
        print("🧹 Patterns cleaned up, \(compositionalPatterns.patterns.count) remaining")
    }
    
    /// Get memory usage estimate
    func estimateMemoryUsage() -> (examples: Int, patterns: Int, segments: Int) {
        let exampleCount = trainingExamples.count
        let patternCount = compositionalPatterns.patterns.count
        let segmentCount = compositionalPatterns.patterns.reduce(0) { $0 + $1.contributingSegments.count }
        return (exampleCount, patternCount, segmentCount)
    }
    
    // MARK: - Development Helpers
    
    func clearAllTrainingData() {
        trainingExamples = []
        compositionalPatterns = PatternCollection()
        userDefaults.removeObject(forKey: trainingExamplesKey)
        userDefaults.removeObject(forKey: compositionalPatternsKey)
        patternDiscoveryEnabled = true // Reset to enabled
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
