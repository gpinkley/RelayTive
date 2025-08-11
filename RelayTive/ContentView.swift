//
//  ContentView.swift
//  RelayTive
//
//  Main tab container for the app
//

import SwiftUI
import Speech
import AVFoundation

struct ContentView: View {
    @StateObject private var dataManager = DataManager()
    @StateObject private var audioManager = AudioManager()
    @StateObject private var translationEngine = TranslationEngine()
    @StateObject private var speechService = SpeechService()
    @StateObject private var captureCoordinator: CaptureCoordinator
    
    @State private var showOnboarding = false
    @State private var hasCompletedOnboarding = false
    
    init() {
        let audioManager = AudioManager()
        let speechService = SpeechService()
        _audioManager = StateObject(wrappedValue: audioManager)
        _speechService = StateObject(wrappedValue: speechService)
        _captureCoordinator = StateObject(wrappedValue: CaptureCoordinator(audioManager: audioManager, speechService: speechService))
        _dataManager = StateObject(wrappedValue: DataManager())
        _translationEngine = StateObject(wrappedValue: TranslationEngine())
    }
    
    var body: some View {
        Group {
            if showOnboarding {
                OnboardingView {
                    completeOnboarding()
                }
            } else {
                TabView {
                    TranslationView()
                        .tabItem {
                            Image(systemName: "mic.badge.plus")
                            Text("Translation")
                        }
                    
                    TrainingView()
                        .tabItem {
                            Image(systemName: "person.wave.2")
                            Text("Training")
                        }
                    
                    ExamplesView()
                        .tabItem {
                            Image(systemName: "list.bullet.clipboard")
                            Text("Examples")
                        }
                }
                .environmentObject(dataManager)
                .environmentObject(audioManager)
                .environmentObject(translationEngine)
                .environmentObject(speechService)
                .environmentObject(captureCoordinator)
            }
        }
        .onAppear {
            checkIfOnboardingNeeded()
        }
    }
    
    // MARK: - Onboarding Logic
    
    private func checkIfOnboardingNeeded() {
        // Check if user has completed onboarding before
        let hasCompletedBefore = UserDefaults.standard.bool(forKey: "HasCompletedOnboarding")
        
        if hasCompletedBefore {
            // Already onboarded, but check if permissions are still granted
            checkExistingPermissions()
        } else {
            // First launch - show onboarding
            showOnboarding = true
        }
    }
    
    private func checkExistingPermissions() {
        Task {
            let micGranted = await checkMicrophonePermission()
            let speechGranted = checkSpeechPermission()
            
            await MainActor.run {
                if !micGranted || !speechGranted {
                    // Permissions were revoked - show onboarding again
                    showOnboarding = true
                } else {
                    hasCompletedOnboarding = true
                }
            }
        }
    }
    
    private func completeOnboarding() {
        UserDefaults.standard.set(true, forKey: "HasCompletedOnboarding")
        hasCompletedOnboarding = true
        showOnboarding = false
    }
    
    // MARK: - Permission Checks
    
    private func checkMicrophonePermission() async -> Bool {
        if #available(iOS 17.0, *) {
            return AVAudioApplication.shared.recordPermission == .granted
        } else {
            return AVAudioSession.sharedInstance().recordPermission == .granted
        }
    }
    
    private func checkSpeechPermission() -> Bool {
        return SFSpeechRecognizer.authorizationStatus() == .authorized
    }
}

#Preview {
    ContentView()
}
