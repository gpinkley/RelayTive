//
//  TrainingView.swift
//  RelayTive
//
//  Training tab - record atypical speech and pair with caregiver explanations
//

import SwiftUI

enum TrainingStep {
    case ready
    case recordingAtypical
    case atypicalRecorded
    case gettingExplanation
    case recordingExplanation
    case complete
}

struct TrainingView: View {
    @EnvironmentObject var audioManager: AudioManager
    @EnvironmentObject var dataManager: DataManager
    @EnvironmentObject var translationEngine: TranslationEngine
    @StateObject private var speechRecognizer = SpeechRecognitionManager()
    
    @State private var currentStep: TrainingStep = .ready
    @State private var atypicalRecording: Data?
    @State private var explanationText = ""
    @State private var showingManualEntry = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 8) {
                    Text("Training")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Record atypical speech, then provide the meaning")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top)
                
                // Progress indicator
                TrainingProgressView(currentStep: currentStep)
                
                // Main content area
                VStack(spacing: 25) {
                    switch currentStep {
                    case .ready:
                        ReadyToRecordView {
                            startAtypicalRecording()
                        }
                        
                    case .recordingAtypical:
                        RecordingAtypicalView {
                            stopAtypicalRecording()
                        }
                        
                    case .atypicalRecorded, .gettingExplanation:
                        RecordedAtypicalView(
                            audioData: atypicalRecording,
                            audioManager: audioManager,
                            onContinue: {
                                currentStep = .gettingExplanation
                                showingManualEntry = false
                            },
                            onManualEntry: {
                                showingManualEntry = true
                            }
                        )
                        
                    case .recordingExplanation:
                        RecordingExplanationView(
                            recognizedText: speechRecognizer.recognizedText,
                            onStop: {
                                stopExplanationRecording()
                            }
                        )
                        
                    case .complete:
                        CompletedTrainingView(
                            explanationText: explanationText,
                            onSave: { editedText in
                                explanationText = editedText
                                saveTrainingExample()
                            },
                            onStartOver: {
                                resetTraining()
                            }
                        )
                    }
                    
                    // Explanation input for gettingExplanation step
                    if currentStep == .gettingExplanation {
                        ExplanationInputView(
                            speechRecognizer: speechRecognizer,
                            explanationText: $explanationText,
                            onSpeechStart: {
                                startExplanationRecording()
                            },
                            onComplete: {
                                currentStep = .complete
                            }
                        )
                    }
                }
                
                Spacer()
            }
            .padding()
        }
        .sheet(isPresented: $showingManualEntry) {
            ManualExplanationEntrySheet(
                explanationText: $explanationText,
                onSave: {
                    showingManualEntry = false
                    currentStep = .complete
                }
            )
        }
    }
    
    // MARK: - Training Flow Actions
    
    private func startAtypicalRecording() {
        currentStep = .recordingAtypical
        audioManager.startRecording()
    }
    
    private func stopAtypicalRecording() {
        atypicalRecording = audioManager.stopRecording()
        currentStep = .atypicalRecorded
    }
    
    private func startExplanationRecording() {
        currentStep = .recordingExplanation
        speechRecognizer.clearText()
        
        Task {
            await speechRecognizer.startRecording()
        }
    }
    
    private func stopExplanationRecording() {
        speechRecognizer.stopRecording()
        explanationText = speechRecognizer.recognizedText
        currentStep = .complete
    }
    
    private func saveTrainingExample() {
        guard let recording = atypicalRecording, !explanationText.isEmpty else { return }
        
        // Extract HuBERT embeddings from the atypical audio BEFORE saving
        Task {
            print("Extracting HuBERT embeddings for new training example...")
            
            guard let embeddings = await translationEngine.extractEmbeddings(recording) else {
                print("❌ Failed to extract embeddings for training example")
                await MainActor.run {
                    // Save without embeddings for now, but this should be improved
                    let utterance = Utterance(
                        originalAudio: recording,
                        translation: explanationText,
                        timestamp: Date(),
                        isVerified: false
                    )
                    dataManager.addUtterance(utterance)
                    
                    // Trigger compositional pattern discovery even without embeddings
                    Task {
                        await dataManager.performCompositionalPatternDiscovery(using: translationEngine)
                    }
                    
                    resetTraining()
                }
                return
            }
            
            print("✅ Successfully extracted \\(embeddings.count) embedding dimensions")
            
            // Create TrainingExample with pre-computed embeddings
            await MainActor.run {
                var trainingExample = TrainingExample(
                    atypicalAudio: recording,
                    typicalExplanation: explanationText,
                    timestamp: Date(),
                    isVerified: false
                )
                trainingExample.setEmbeddings(embeddings)
                
                dataManager.addTrainingExample(trainingExample)
                
                // Trigger compositional pattern discovery after adding new training data
                Task {
                    await dataManager.performCompositionalPatternDiscovery(using: translationEngine)
                }
                
                resetTraining()
            }
        }
    }
    
    private func resetTraining() {
        currentStep = .ready
        atypicalRecording = nil
        explanationText = ""
        speechRecognizer.clearText()
        showingManualEntry = false
    }
}

// MARK: - Training Progress View

struct TrainingProgressView: View {
    let currentStep: TrainingStep
    
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<4, id: \.self) { index in
                Circle()
                    .fill(stepColor(for: index))
                    .frame(width: 12, height: 12)
            }
        }
        .padding()
    }
    
    private func stepColor(for index: Int) -> Color {
        let stepIndex = stepToIndex(currentStep)
        return index <= stepIndex ? .blue : .gray.opacity(0.3)
    }
    
    private func stepToIndex(_ step: TrainingStep) -> Int {
        switch step {
        case .ready: return -1
        case .recordingAtypical: return 0
        case .atypicalRecorded: return 1
        case .gettingExplanation: return 2
        case .recordingExplanation: return 2
        case .complete: return 3
        }
    }
}

// MARK: - Step Views

struct ReadyToRecordView: View {
    let onRecord: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "mic.circle")
                .font(.system(size: 80))
                .foregroundColor(.blue)
            
            Text("Ready to Record")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Press to record an atypical speech sample")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Button(action: onRecord) {
                Text("Start Recording")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(12)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
    }
}

struct RecordingAtypicalView: View {
    let onStop: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Circle()
                .fill(Color.red)
                .frame(width: 100, height: 100)
                .overlay(
                    Image(systemName: "mic.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.white)
                )
                .scaleEffect(1.1)
                .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: UUID())
            
            Text("Recording Atypical Speech...")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Speak naturally in your unique way")
                .font(.body)
                .foregroundColor(.secondary)
            
            Button(action: onStop) {
                Text("Stop Recording")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .cornerRadius(12)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
    }
}

struct RecordedAtypicalView: View {
    let audioData: Data?
    let audioManager: AudioManager
    let onContinue: () -> Void
    let onManualEntry: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)
            
            Text("Atypical Speech Recorded")
                .font(.title2)
                .fontWeight(.semibold)
            
            // Playback button
            if let audioData = audioData {
                Button(action: {
                    audioManager.playAudio(audioData)
                }) {
                    HStack {
                        Image(systemName: "play.circle.fill")
                        Text("Play Recording")
                    }
                    .font(.body)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.blue.opacity(0.1))
                    .foregroundColor(.blue)
                    .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
            }
            
            Text("Now provide what this means in typical language")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            HStack(spacing: 15) {
                Button(action: onContinue) {
                    VStack {
                        Image(systemName: "mic.circle")
                            .font(.title)
                        Text("Speak")
                            .font(.caption)
                    }
                    .foregroundColor(.blue)
                    .frame(width: 80, height: 60)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                }
                .buttonStyle(PlainButtonStyle())
                
                Button(action: onManualEntry) {
                    VStack {
                        Image(systemName: "keyboard")
                            .font(.title)
                        Text("Type")
                            .font(.caption)
                    }
                    .foregroundColor(.gray)
                    .frame(width: 80, height: 60)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                }
                .buttonStyle(PlainButtonStyle())
            }
        }
        .padding()
    }
}

struct RecordingExplanationView: View {
    let recognizedText: String
    let onStop: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Circle()
                .fill(Color.orange)
                .frame(width: 80, height: 80)
                .overlay(
                    Image(systemName: "waveform")
                        .font(.system(size: 30))
                        .foregroundColor(.white)
                )
            
            Text("Listening for Explanation...")
                .font(.title2)
                .fontWeight(.semibold)
            
            // Show recognized text
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
                .frame(minHeight: 100)
                .overlay(
                    Text(recognizedText.isEmpty ? "Your explanation will appear here..." : recognizedText)
                        .font(.body)
                        .foregroundColor(recognizedText.isEmpty ? .secondary : .primary)
                        .multilineTextAlignment(.leading)
                        .padding()
                )
            
            Button(action: onStop) {
                Text("Done Speaking")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.orange)
                    .cornerRadius(12)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
    }
}

struct CompletedTrainingView: View {
    @State private var editableExplanationText: String
    let onSave: (String) -> Void
    let onStartOver: () -> Void
    
    init(explanationText: String, onSave: @escaping (String) -> Void, onStartOver: @escaping () -> Void) {
        self._editableExplanationText = State(initialValue: explanationText)
        self.onSave = onSave
        self.onStartOver = onStartOver
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "star.circle.fill")
                .font(.system(size: 60))
                .foregroundColor(.green)
            
            Text("Training Example Complete")
                .font(.title2)
                .fontWeight(.semibold)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Explanation:")
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Text("Edit if needed before saving:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                TextEditor(text: $editableExplanationText)
                    .font(.body)
                    .padding(8)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    .frame(minHeight: 80)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(.systemGray4), lineWidth: 1)
                    )
            }
            
            HStack(spacing: 15) {
                Button(action: onStartOver) {
                    Text("Record Another")
                        .font(.body)
                        .foregroundColor(.blue)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(10)
                }
                .buttonStyle(PlainButtonStyle())
                
                Button(action: { onSave(editableExplanationText) }) {
                    Text("Save Example")
                        .font(.body)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
                .buttonStyle(PlainButtonStyle())
            }
        }
        .padding()
        .contentShape(Rectangle())
        .onTapGesture {
            // Dismiss keyboard when tapping outside
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }
}

struct ExplanationInputView: View {
    @ObservedObject var speechRecognizer: SpeechRecognitionManager
    @Binding var explanationText: String
    let onSpeechStart: () -> Void
    let onComplete: () -> Void
    
    var body: some View {
        VStack(spacing: 15) {
            HStack(spacing: 15) {
                Button(action: onSpeechStart) {
                    VStack {
                        Image(systemName: "mic.circle.fill")
                            .font(.title)
                        Text("Record")
                            .font(.caption)
                    }
                    .foregroundColor(speechRecognizer.canRecord ? .blue : .gray)
                    .frame(width: 80, height: 60)
                    .background((speechRecognizer.canRecord ? Color.blue : Color.gray).opacity(0.1))
                    .cornerRadius(10)
                }
                .disabled(!speechRecognizer.canRecord)
                .buttonStyle(PlainButtonStyle())
                
                VStack(alignment: .leading, spacing: 5) {
                    Text("Or type manually:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    TextField("Enter explanation...", text: $explanationText, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(3...6)
                }
            }
            
            if !explanationText.isEmpty {
                Button(action: onComplete) {
                    Text("Continue")
                        .font(.body)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
                .buttonStyle(PlainButtonStyle())
            }
            
            if let error = speechRecognizer.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .contentShape(Rectangle())
        .onTapGesture {
            // Dismiss keyboard when tapping outside
            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
        }
    }
}

struct ManualExplanationEntrySheet: View {
    @Binding var explanationText: String
    let onSave: () -> Void
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Enter Explanation")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Type what the atypical speech means in clear, typical language:")
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                
                TextEditor(text: $explanationText)
                    .font(.body)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    .frame(minHeight: 120)
                
                Spacer()
            }
            .padding()
            .contentShape(Rectangle())
            .onTapGesture {
                // Dismiss keyboard when tapping outside
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
            .navigationTitle("Manual Entry")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        onSave()
                        dismiss()
                    }
                    .disabled(explanationText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .fontWeight(.semibold)
                }
            }
        }
    }
}

#Preview {
    TrainingView()
        .environmentObject(DataManager())
        .environmentObject(AudioManager())
        .environmentObject(TranslationEngine())
}