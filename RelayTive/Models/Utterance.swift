//
//  Utterance.swift
//  RelayTive
//
//  Core data model for storing audio recordings and their translations
//

import Foundation

struct Utterance: Identifiable, Codable {
    let id: UUID
    let originalAudio: Data
    var translation: String
    let timestamp: Date
    var isVerified: Bool
    
    init(originalAudio: Data, translation: String, timestamp: Date, isVerified: Bool = false) {
        self.id = UUID()
        self.originalAudio = originalAudio
        self.translation = translation
        self.timestamp = timestamp
        self.isVerified = isVerified
    }
    
    init(id: UUID, originalAudio: Data, translation: String, timestamp: Date, isVerified: Bool = false) {
        self.id = id
        self.originalAudio = originalAudio
        self.translation = translation
        self.timestamp = timestamp
        self.isVerified = isVerified
    }
}

// MARK: - Convenience Extensions
extension Utterance {
    var isRecent: Bool {
        let oneHourAgo = Date().addingTimeInterval(-3600)
        return timestamp > oneHourAgo
    }
    
    var hasAudio: Bool {
        return !originalAudio.isEmpty
    }
}

// MARK: - Sample Data
extension Utterance {
    static let sampleData: [Utterance] = [
        Utterance(
            originalAudio: Data(),
            translation: "I want some water",
            timestamp: Date().addingTimeInterval(-300),
            isVerified: true
        ),
        Utterance(
            originalAudio: Data(),
            translation: "Help me please",
            timestamp: Date().addingTimeInterval(-600),
            isVerified: false
        ),
        Utterance(
            originalAudio: Data(),
            translation: "I need to go outside",
            timestamp: Date().addingTimeInterval(-900),
            isVerified: true
        )
    ]
}