//
//  TranslationView.swift
//  RelayTive
//
//  Main translation tab - real-time speech translation
//

import SwiftUI

struct TranslationView: View {
    @EnvironmentObject var audioManager: AudioManager
    @EnvironmentObject var dataManager: DataManager
    @EnvironmentObject var translationEngine: TranslationEngine
    @State private var isRecording = false
    @State private var currentTranslation = ""
    
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
                
                Spacer()
                
                // Recent translations history
                if !dataManager.recentTranslations.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Recent Translations")
                            .font(.headline)
                        
                        ScrollView {
                            LazyVStack(spacing: 8) {
                                ForEach(dataManager.recentTranslations.prefix(5)) { utterance in
                                    RecentTranslationRow(utterance: utterance)
                                }
                            }
                        }
                        .frame(maxHeight: 150)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
            }
            .padding()
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
            return
        }
        
        // Process audio through HuBERT translation engine
        Task {
            if let translation = await translationEngine.translateAudio(audioData) {
                await MainActor.run {
                    currentTranslation = translation
                    
                    // Save the translation
                    let utterance = Utterance(
                        originalAudio: audioData,
                        translation: translation,
                        timestamp: Date(),
                        isVerified: false
                    )
                    dataManager.addTranslation(utterance)
                }
            } else {
                await MainActor.run {
                    currentTranslation = "Translation failed. Please try again."
                }
            }
        }
    }
}

struct RecentTranslationRow: View {
    let utterance: Utterance
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(utterance.translation)
                    .font(.body)
                    .foregroundColor(.primary)
                
                Text(utterance.timestamp, style: .time)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: utterance.isVerified ? "checkmark.circle.fill" : "circle")
                .foregroundColor(utterance.isVerified ? .green : .gray)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .cornerRadius(8)
    }
}

#Preview {
    TranslationView()
        .environmentObject(DataManager())
        .environmentObject(AudioManager())
        .environmentObject(TranslationEngine())
}