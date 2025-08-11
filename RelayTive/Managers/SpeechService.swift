//
//  SpeechService.swift
//  RelayTive
//
//  Robust speech recognition with proper teardown and 1101 error handling
//

import Foundation
import Speech
import AVFoundation
import SwiftUI

@MainActor
class SpeechService: ObservableObject {
    @Published var recognizedText = ""
    @Published var isRecording = false
    @Published var errorMessage: String?
    @Published var isAvailable = false
    
    // Speech components (all must be properly torn down)
    private var recognizer: SFSpeechRecognizer?
    private var task: SFSpeechRecognitionTask?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private let engine = AVAudioEngine()
    
    // Error backoff for 1101 errors
    private var lastError1101Time: Date?
    private var error1101CooldownSeconds: TimeInterval = 0.4
    private var hasLoggedResetRecently = false
    private let resetLogCooldownSeconds: TimeInterval = 2.0
    
    // Delegate reference
    private var speechRecognizerDelegate: SpeechRecognizerDelegate?
    
    init(locale: Locale = Locale.current) {
        setupRecognizer(locale: locale)
    }
    
    // MARK: - Setup
    
    private func setupRecognizer(locale: Locale) {
        recognizer = SFSpeechRecognizer(locale: locale)
        isAvailable = recognizer?.isAvailable ?? false
        
        // Monitor availability changes
        speechRecognizerDelegate = SpeechRecognizerDelegate { [weak self] available in
            Task { @MainActor in
                self?.isAvailable = available
            }
        }
        recognizer?.delegate = speechRecognizerDelegate
        
        print("🔧 SpeechService: Recognizer setup for locale \(locale.identifier), available: \(isAvailable)")
    }
    
    // MARK: - Authorization Check
    
    private func checkAuthorization() -> Bool {
        let speechStatus = SFSpeechRecognizer.authorizationStatus()
        guard speechStatus == .authorized else {
            errorMessage = "Speech recognition not authorized"
            print("❌ SpeechService: Speech authorization status: \(speechStatus)")
            return false
        }
        return true
    }
    
    // MARK: - Start Recognition
    
    func startRecognition(locale: Locale = Locale.current) async {
        print("🎯 SpeechService: startRecognition() - isRecording: \(isRecording)")
        
        // Preflight authorization check
        guard checkAuthorization() else { return }
        
        // Check 1101 error cooldown
        if let lastError = lastError1101Time,
           Date().timeIntervalSince(lastError) < error1101CooldownSeconds {
            print("⏳ SpeechService: Still in 1101 error cooldown")
            return
        }
        
        // Setup recognizer for locale if needed
        if recognizer?.locale != locale {
            setupRecognizer(locale: locale)
        }
        
        guard let rec = recognizer, rec.isAvailable else {
            errorMessage = "Speech recognizer not available"
            return
        }
        
        // Stop any existing recognition
        await stopRecognition(reason: "starting new recognition")
        
        do {
            // Configure audio session (coordinator should have done this already)
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            
            // Create recognition request
            request = SFSpeechAudioBufferRecognitionRequest()
            guard let request = request else {
                errorMessage = "Failed to create recognition request"
                return
            }
            
            // Use on-device recognition when available
            request.requiresOnDeviceRecognition = rec.supportsOnDeviceRecognition
            request.shouldReportPartialResults = true
            
            // Install input tap BEFORE starting engine
            let inputNode = engine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
                self?.request?.append(buffer)
            }
            
            // Start engine before creating task
            engine.prepare()
            try engine.start()
            
            // Create recognition task
            task = rec.recognitionTask(with: request) { [weak self] result, error in
                Task { @MainActor in
                    if let result = result {
                        self?.recognizedText = result.bestTranscription.formattedString
                    }
                    
                    if let error = error {
                        await self?.handleRecognitionError(error)
                    }
                    
                    if result?.isFinal == true {
                        await self?.stopRecognition(reason: "final result received")
                    }
                }
            }
            
            isRecording = true
            recognizedText = ""
            errorMessage = nil
            
            let onDeviceStatus = request.requiresOnDeviceRecognition ? "true" : "false"
            print("🎙️ Speech start (onDevice: \(onDeviceStatus))")
            
        } catch {
            errorMessage = "Failed to start speech recognition: \(error.localizedDescription)"
            await stopRecognition(reason: "start failed")
        }
    }
    
    // MARK: - Stop Recognition (idempotent)
    
    func stopRecognition(reason: String) async {
        print("🛑 Speech stop (reason: \(reason))")
        
        // End audio request first
        request?.endAudio()
        
        // Cancel and clear task
        task?.cancel()
        task = nil
        
        // Remove input tap if present and stop engine
        if engine.inputNode.numberOfInputs > 0 {
            engine.inputNode.removeTap(onBus: 0)
        }
        
        if engine.isRunning {
            engine.stop()
            engine.reset()
        }
        
        // Deactivate session
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("⚠️ Failed to deactivate audio session: \(error)")
        }
        
        // Clear request and recognizer references
        request = nil
        recognizer = nil
        
        isRecording = false
    }
    
    // MARK: - Error Handling with 1101 Backoff
    
    private func handleRecognitionError(_ error: Error) async {
        let nsError = error as NSError
        
        // Handle 1101 errors with backoff and reset
        if nsError.code == 1101 {
            lastError1101Time = Date()
            
            // Log reset message with cooldown to prevent spam
            let now = Date()
            if let lastResetLog = lastError1101Time,
               now.timeIntervalSince(lastResetLog) > resetLogCooldownSeconds || !hasLoggedResetRecently {
                print("⚠️ Speech error 1101: resetting pipeline")
                hasLoggedResetRecently = true
                
                // Auto-reset after cooldown
                DispatchQueue.main.asyncAfter(deadline: .now() + resetLogCooldownSeconds) {
                    self.hasLoggedResetRecently = false
                }
            }
            
            // Perform full teardown and wait before allowing retry
            await stopRecognition(reason: "1101 error")
            
            // Brief delay before allowing retry
            try? await Task.sleep(nanoseconds: UInt64(error1101CooldownSeconds * 1_000_000_000))
            
            // Reset recognizer to fresh state
            setupRecognizer(locale: recognizer?.locale ?? Locale.current)
            
            errorMessage = "Speech service reset due to system conflict"
        } else {
            errorMessage = "Recognition error: \(error.localizedDescription)"
            print("🔥 Speech error \(nsError.code): \(error.localizedDescription)")
        }
        
        await stopRecognition(reason: "error occurred")
    }
    
    // MARK: - Public Utilities
    
    var canRecord: Bool {
        return isAvailable && SFSpeechRecognizer.authorizationStatus() == .authorized
    }
    
    func clearText() {
        recognizedText = ""
        errorMessage = nil
    }
    
    func resetErrorState() {
        lastError1101Time = nil
        hasLoggedResetRecently = false
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