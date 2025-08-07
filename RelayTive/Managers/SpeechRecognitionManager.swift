//
//  SpeechRecognitionManager.swift
//  RelayTive
//
//  Real iOS Speech Recognition for caregiver explanations
//

import Foundation
import Speech
import AVFoundation
import SwiftUI

@MainActor
class SpeechRecognitionManager: ObservableObject {
    @Published var isRecording = false
    @Published var isAvailable = false
    @Published var recognizedText = ""
    @Published var errorMessage: String?
    
    private let speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var speechRecognizerDelegate: SpeechRecognizerDelegate?
    
    init() {
        // Use device locale, fallback to English
        speechRecognizer = SFSpeechRecognizer(locale: Locale.current) ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        
        setupSpeechRecognition()
    }
    
    // MARK: - Setup
    
    private func setupSpeechRecognition() {
        guard let speechRecognizer = speechRecognizer else {
            errorMessage = "Speech recognition not available for this locale"
            return
        }
        
        isAvailable = speechRecognizer.isAvailable
        
        // Monitor availability changes with strong reference to delegate
        speechRecognizerDelegate = SpeechRecognizerDelegate { [weak self] available in
            Task { @MainActor in
                self?.isAvailable = available
                if !available {
                    self?.errorMessage = "Speech recognition became unavailable"
                }
            }
        }
        speechRecognizer.delegate = speechRecognizerDelegate
    }
    
    // MARK: - Permissions
    
    func requestPermissions() async -> Bool {
        // Request speech recognition authorization
        let speechStatus = await requestSpeechAuthorization()
        guard speechStatus == .authorized else {
            await MainActor.run {
                errorMessage = "Speech recognition permission denied"
            }
            return false
        }
        
        // Request microphone authorization
        let audioStatus = await requestMicrophoneAuthorization()
        guard audioStatus else {
            await MainActor.run {
                errorMessage = "Microphone permission denied"
            }
            return false
        }
        
        await MainActor.run {
            errorMessage = nil
        }
        return true
    }
    
    private func requestSpeechAuthorization() async -> SFSpeechRecognizerAuthorizationStatus {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
    }
    
    private func requestMicrophoneAuthorization() async -> Bool {
        if #available(iOS 17.0, *) {
            return await withCheckedContinuation { continuation in
                AVAudioApplication.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        } else {
            return await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        }
    }
    
    // MARK: - Speech Recognition
    
    func startRecording() async {
        // Ensure we have permissions
        guard await requestPermissions() else { return }
        
        // Ensure speech recognizer is available
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            errorMessage = "Speech recognition not available"
            return
        }
        
        // Cancel any existing task
        stopRecording()
        
        do {
            // Configure audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
            
            // Create recognition request
            recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
            guard let recognitionRequest = recognitionRequest else {
                errorMessage = "Unable to create recognition request"
                return
            }
            
            recognitionRequest.shouldReportPartialResults = true
            
            // Create recognition task
            recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                Task { @MainActor in
                    if let result = result {
                        self?.recognizedText = result.bestTranscription.formattedString
                    }
                    
                    if let error = error {
                        self?.errorMessage = "Recognition error: \(error.localizedDescription)"
                        self?.stopRecording()
                    }
                    
                    // Final result received
                    if result?.isFinal == true {
                        self?.stopRecording()
                    }
                }
            }
            
            // Start audio engine
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                recognitionRequest.append(buffer)
            }
            
            audioEngine.prepare()
            try audioEngine.start()
            
            isRecording = true
            recognizedText = ""
            errorMessage = nil
            print("Speech recognition started")
            
        } catch {
            errorMessage = "Failed to start speech recognition: \(error.localizedDescription)"
            stopRecording()
        }
    }
    
    func stopRecording() {
        // Stop audio engine
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        // Finish recognition request
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        
        // Cancel recognition task
        recognitionTask?.cancel()
        recognitionTask = nil
        
        isRecording = false
        print("Speech recognition stopped")
        
        // Reset audio session
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to reset audio session: \(error)")
        }
    }
    
    // MARK: - Utilities
    
    var canRecord: Bool {
        return isAvailable && speechRecognizer != nil
    }
    
    func clearText() {
        recognizedText = ""
        errorMessage = nil
    }
}

// MARK: - Speech Recognizer Delegate Helper

private class SpeechRecognizerDelegate: NSObject, SFSpeechRecognizerDelegate {
    private let onAvailabilityChange: (Bool) -> Void
    
    init(onAvailabilityChange: @escaping (Bool) -> Void) {
        self.onAvailabilityChange = onAvailabilityChange
    }
    
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        onAvailabilityChange(available)
    }
}