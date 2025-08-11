//
//  ContentView.swift
//  RelayTive
//
//  Main tab container for the app
//

import SwiftUI

struct ContentView: View {
    @StateObject private var dataManager = DataManager()
    @StateObject private var audioManager = AudioManager()
    @StateObject private var translationEngine = TranslationEngine()
    @StateObject private var speechService = SpeechService()
    @StateObject private var captureCoordinator: CaptureCoordinator
    
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

#Preview {
    ContentView()
}
