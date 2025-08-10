//
//  TranslationView.swift
//  RelayTive
//
//  Main translation tab - real-time speech translation
//

import SwiftUI
import UniformTypeIdentifiers

struct TranslationView: View {
    @EnvironmentObject var audioManager: AudioManager
    @EnvironmentObject var dataManager: DataManager
    @EnvironmentObject var translationEngine: TranslationEngine
    @State private var isRecording = false
    @State private var currentTranslation = ""
    @State private var showingDocumentPicker = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 8) {
                    Text("Translation")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Tap to translate atypical speech")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top)
                
                Spacer()
                
                // Current translation result
                VStack(spacing: 16) {
                    Text("Translation Result")
                        .font(.headline)
                    
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(.systemGray6))
                        .frame(height: 120)
                        .overlay(
                            Text(currentTranslation.isEmpty ? "Translation will appear here..." : currentTranslation)
                                .font(.title2)
                                .foregroundColor(currentTranslation.isEmpty ? .secondary : .primary)
                                .multilineTextAlignment(.center)
                                .padding()
                        )
                }
                
                // Record button
                Button(action: toggleRecording) {
                    Circle()
                        .fill(isRecording ? Color.red : Color.blue)
                        .frame(width: 120, height: 120)
                        .overlay(
                            Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 40))
                                .foregroundColor(.white)
                        )
                        .scaleEffect(isRecording ? 1.1 : 1.0)
                        .animation(.easeInOut(duration: 0.1), value: isRecording)
                }
                .buttonStyle(PlainButtonStyle())
                
                Text(isRecording ? "Recording..." : "Tap to Record")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                // Debug audio file importer
                #if DEBUG
                VStack(spacing: 12) {
                    Text("Debug File Importer")
                        .font(.headline)
                    
                    Button("Import Audio File (WAV/CAF)") {
                        showingDocumentPicker = true
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .background(Color(.systemYellow).opacity(0.1))
                .cornerRadius(12)
                #endif
                
                Spacer()
                
                // Training examples info (read-only)
                if dataManager.totalTrainingExamples > 0 {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Available Training Examples")
                            .font(.headline)
                        
                        HStack {
                            VStack(alignment: .leading) {
                                Text("\(dataManager.totalTrainingExamples) trained phrases")
                                    .font(.body)
                                Text("\(Int(dataManager.embeddingsCompletionRate * 100))% processed")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Text("Add more in Training tab")
                                .font(.caption)
                                .foregroundColor(.blue)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
            }
            .padding()
        }
        .fileImporter(
            isPresented: $showingDocumentPicker,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            handleFileImport(result: result)
        }
    }
    
    private func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }
    
    private func startRecording() {
        isRecording = true
        audioManager.startRecording()
        currentTranslation = "" // Clear previous translation
    }
    
    private func stopRecording() {
        isRecording = false
        
        // Get recorded audio data
        guard let audioData = audioManager.stopRecording() else {
            print("No audio data captured")
            currentTranslation = "No audio captured. Please try again."
            return
        }
        
        // Process audio through HuBERT model to get embeddings, then lookup in training data
        Task {
            // Step 1: Extract embeddings using HuBERT model
            guard translationEngine.isModelLoaded else {
                await MainActor.run {
                    currentTranslation = "HuBERT model not loaded. Please check model files."
                }
                return
            }
            
            // Get embeddings from the TranslationEngine (but not translation)
            let embeddings = await extractEmbeddingsFromAudio(audioData)
            
            guard let embeddings = embeddings else {
                await MainActor.run {
                    currentTranslation = "Failed to process audio. Please try again."
                }
                return
            }
            
            // Step 2: Look up translation using compositional pattern matching with fallback
            if let match = await dataManager.findTranslationForAudio(audioData, embeddings: embeddings, using: translationEngine) {
                await MainActor.run {
                    let confidencePercent = Int(match.confidence * 100)
                    currentTranslation = "\(match.translation) (\(confidencePercent)% match)"
                    print("Translation found: \(match.translation) with \(confidencePercent)% confidence")
                }
            } else {
                await MainActor.run {
                    currentTranslation = "No matching training example found. Please add this phrase in the Training tab first."
                }
            }
            
            // NOTE: We do NOT save temporary translations - Translation tab is for lookup only
        }
    }
    
    // Helper function to extract embeddings using the TranslationEngine
    private func extractEmbeddingsFromAudio(_ audioData: Data) async -> [Float]? {
        // Use the new extractEmbeddings method from TranslationEngine
        return await translationEngine.extractEmbeddings(audioData)
    }
    
    #if DEBUG
    private func handleFileImport(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            processImportedAudioFile(url: url)
        case .failure(let error):
            print("❌ File import failed: \(error)")
            currentTranslation = "File import failed"
        }
    }
    
    private func processImportedAudioFile(url: URL) {
        Task {
            do {
                let audioData = try Data(contentsOf: url)
                print("🔍 Processing imported file: \(url.lastPathComponent) (\(audioData.count) bytes)")
                
                // Run through the same translation path as live recording
                guard translationEngine.isModelLoaded else {
                    await MainActor.run {
                        currentTranslation = "HuBERT model not loaded"
                    }
                    return
                }
                
                let embeddings = await extractEmbeddingsFromAudio(audioData)
                
                guard let embeddings = embeddings else {
                    await MainActor.run {
                        currentTranslation = "Failed to process audio file"
                    }
                    return
                }
                
                if let match = await dataManager.findTranslationForAudio(audioData, embeddings: embeddings, using: translationEngine) {
                    await MainActor.run {
                        currentTranslation = match.translation
                    }
                } else {
                    await MainActor.run {
                        currentTranslation = "No matching pattern found for imported file"
                    }
                }
            } catch {
                print("❌ Error reading audio file: \(error)")
                await MainActor.run {
                    currentTranslation = "Error reading audio file"
                }
            }
        }
    }
    #endif
}


#Preview {
    TranslationView()
        .environmentObject(DataManager())
        .environmentObject(AudioManager())
        .environmentObject(TranslationEngine())
}