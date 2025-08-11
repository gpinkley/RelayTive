//
//  CaptureStateMachine.swift
//  RelayTive
//
//  Central coordinator for audio capture to prevent simultaneous recording/speech recognition
//

import Foundation
import AVFoundation
import SwiftUI

enum CaptureState: String {
    case idle
    case recordingAppAudio  
    case speechRecognition
}

@MainActor
final class CaptureCoordinator: ObservableObject {
    @Published var state: CaptureState = .idle
    
    // Dependencies (injected)
    private weak var audioManager: AudioManager?
    private weak var speechService: SpeechService?
    
    // Session management
    private let session = AVAudioSession.sharedInstance()
    
    init(audioManager: AudioManager? = nil, speechService: SpeechService? = nil) {
        self.audioManager = audioManager
        self.speechService = speechService
    }
    
    // MARK: - App Recording Coordination
    
    func beginAppRecording() async {
        print("🎯 CaptureCoordinator: beginAppRecording() - current state: \(state)")
        
        // Stop speech recognition if running
        if state == .speechRecognition {
            await endSpeechRecognition(reason: "switching to app recording")
        }
        
        guard state == .idle else {
            print("⚠️ CaptureCoordinator: Cannot start app recording, state is \(state)")
            return
        }
        
        // Configure session for our app recording
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            
            state = .recordingAppAudio
            print("🎙️ App recording session configured (category: record, mode: measurement)")
            
            // Now start actual recording through AudioManager
            audioManager?.startRecording()
            
        } catch {
            print("❌ Failed to configure session for app recording: \(error)")
        }
    }
    
    func endAppRecording() {
        print("🛑 CaptureCoordinator: endAppRecording() - current state: \(state)")
        
        guard state == .recordingAppAudio else {
            print("⚠️ CaptureCoordinator: Not in app recording state, ignoring")
            return
        }
        
        // Stop recording and reset session
        do {
            try session.setActive(false, options: .notifyOthersOnDeactivation)
            state = .idle
            print("🔇 App recording session deactivated")
        } catch {
            print("⚠️ Failed to deactivate session after app recording: \(error)")
            state = .idle // Force reset state even if session deactivation failed
        }
    }
    
    // MARK: - Speech Recognition Coordination
    
    func beginSpeechRecognition(locale: Locale = Locale.current) async {
        print("🎯 CaptureCoordinator: beginSpeechRecognition() - current state: \(state)")
        
        // Stop app recording if running
        if state == .recordingAppAudio {
            // Stop our recording first
            if let data = audioManager?.stopRecording() {
                print("📱 Stopped app recording (got \(data.count) bytes) to start speech")
            }
            endAppRecording()
        }
        
        guard state == .idle else {
            print("⚠️ CaptureCoordinator: Cannot start speech recognition, state is \(state)")
            return
        }
        
        // Configure session for speech recognition
        do {
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers, .allowBluetooth])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
            
            state = .speechRecognition
            print("🎙️ Speech recognition session configured (category: record, mode: measurement)")
            
            // Start speech recognition through SpeechService
            await speechService?.startRecognition(locale: locale)
            
        } catch {
            print("❌ Failed to configure session for speech recognition: \(error)")
        }
    }
    
    func endSpeechRecognition(reason: String) async {
        print("🛑 CaptureCoordinator: endSpeechRecognition(reason: \(reason)) - current state: \(state)")
        
        guard state == .speechRecognition else {
            print("⚠️ CaptureCoordinator: Not in speech recognition state, ignoring")
            return
        }
        
        // Stop speech recognition
        await speechService?.stopRecognition(reason: reason)
        
        // Reset session
        do {
            try session.setActive(false, options: .notifyOthersOnDeactivation)
            state = .idle
            print("🔇 Speech recognition session deactivated")
        } catch {
            print("⚠️ Failed to deactivate session after speech recognition: \(error)")
            state = .idle // Force reset state even if session deactivation failed
        }
    }
    
    // MARK: - Utilities
    
    var canStartAppRecording: Bool {
        return state == .idle || state == .speechRecognition
    }
    
    var canStartSpeechRecognition: Bool {
        return state == .idle || state == .recordingAppAudio
    }
    
    // Force reset if needed (for error recovery)
    func forceReset() async {
        print("🔄 CaptureCoordinator: forceReset() - current state: \(state)")
        
        // Stop everything
        await speechService?.stopRecognition(reason: "force reset")
        audioManager?.stopPlayback()
        
        // Reset session
        do {
            try session.setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("⚠️ Failed to reset session during force reset: \(error)")
        }
        
        state = .idle
        print("♻️ CaptureCoordinator reset to idle state")
    }
}