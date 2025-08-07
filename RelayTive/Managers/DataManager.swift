//
//  DataManager.swift
//  RelayTive
//
//  Manages local storage and retrieval of utterances and translations
//

import Foundation
import SwiftUI

@MainActor
class DataManager: ObservableObject {
    @Published var allUtterances: [Utterance] = []
    
    private let userDefaults = UserDefaults.standard
    private let utterancesKey = "SavedUtterances"
    
    init() {
        loadUtterances()
    }
    
    // MARK: - Utterance Management
    
    func addUtterance(_ utterance: Utterance) {
        allUtterances.insert(utterance, at: 0) // Add to beginning for most recent first
        saveUtterances()
    }
    
    func updateUtterance(_ updatedUtterance: Utterance) {
        if let index = allUtterances.firstIndex(where: { $0.id == updatedUtterance.id }) {
            allUtterances[index] = updatedUtterance
            saveUtterances()
        }
    }
    
    func removeUtterance(_ utterance: Utterance) {
        allUtterances.removeAll { $0.id == utterance.id }
        saveUtterances()
    }
    
    // MARK: - Translation History
    
    var recentTranslations: [Utterance] {
        return allUtterances.filter(\.isRecent)
    }
    
    func addTranslation(_ utterance: Utterance) {
        addUtterance(utterance)
    }
    
    // MARK: - Verification
    
    var unverifiedUtterances: [Utterance] {
        return allUtterances.filter { !$0.isVerified }
    }
    
    var verifiedUtterances: [Utterance] {
        return allUtterances.filter(\.isVerified)
    }
    
    // MARK: - Statistics
    
    var totalUtterances: Int {
        return allUtterances.count
    }
    
    var verificationRate: Double {
        guard totalUtterances > 0 else { return 0.0 }
        let verifiedCount = verifiedUtterances.count
        return Double(verifiedCount) / Double(totalUtterances)
    }
    
    // MARK: - Persistence
    
    private func saveUtterances() {
        do {
            let data = try JSONEncoder().encode(allUtterances)
            userDefaults.set(data, forKey: utterancesKey)
        } catch {
            print("Failed to save utterances: \(error)")
        }
    }
    
    private func loadUtterances() {
        guard let data = userDefaults.data(forKey: utterancesKey) else {
            // Start with empty dataset - no sample data
            allUtterances = []
            return
        }
        
        do {
            allUtterances = try JSONDecoder().decode([Utterance].self, from: data)
        } catch {
            print("Failed to load utterances: \(error)")
            allUtterances = []
        }
    }
    
    // MARK: - Development Helpers
    
    func clearAllData() {
        allUtterances = []
        userDefaults.removeObject(forKey: utterancesKey)
    }
    
    func loadSampleData() {
        allUtterances = Utterance.sampleData
        saveUtterances()
    }
}