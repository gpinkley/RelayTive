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
    }
}

#Preview {
    ContentView()
}
