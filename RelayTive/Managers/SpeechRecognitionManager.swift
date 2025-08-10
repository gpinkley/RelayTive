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
    
    // Error handling and retry logic
    private var lastErrorTime: Date?
    private var consecutiveErrors: Int = 0
    private let maxConsecutiveErrors = 3
    private let errorCooldownPeriod: TimeInterval = 2.0
    
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
        // Check for recent errors and apply backoff
        if let lastError = lastErrorTime, 
           Date().timeIntervalSince(lastError) < errorCooldownPeriod {
            errorMessage = "Speech recognition cooling down after recent errors"
            return
        }
        
        // Check for too many consecutive errors
        if consecutiveErrors >= maxConsecutiveErrors {
            errorMessage = "Too many consecutive speech recognition errors. Please try again later."
            return
        }
        
        // Ensure we have permissions
        guard await requestPermissions() else { return }
        
        // Ensure speech recognizer is available
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            errorMessage = "Speech recognition not available"
            return
        }
        
        // Cancel any existing task properly
        stopRecording()
        
        // Prevent starting if already running
        guard !isRecording else {
            print("Speech recognition already running")
            return
        }
        
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
                        // Reset error count on successful result
                        self?.consecutiveErrors = 0
                    }
                    
                    if let error = error {
                        let nsError = error as NSError
                        self?.handleRecognitionError(nsError)
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
    
    // MARK: - Error Handling
    
    private func handleRecognitionError(_ error: NSError) {
        lastErrorTime = Date()
        consecutiveErrors += 1
        
        // Handle specific error codes
        switch error.code {
        case 1101: // kAFAssistant error
            errorMessage = "Speech recognition service temporarily unavailable"
            print("Speech recognition error 1101 - service issue, will backoff")
        case 203: // Network error
            errorMessage = "Network error during speech recognition"
        default:
            errorMessage = "Recognition error: \(error.localizedDescription)"
        }
        
        print("Speech recognition error \(error.code): \(error.localizedDescription), consecutive errors: \(consecutiveErrors)")
    }
    
    // Reset error state manually if needed
    func resetErrorState() {
        consecutiveErrors = 0
        lastErrorTime = nil
        errorMessage = nil
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