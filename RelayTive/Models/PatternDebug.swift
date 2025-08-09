//
//  PatternDebug.swift
//  RelayTive
//
//  DEBUG-only utility for pattern analysis and reporting
//

import Foundation

#if DEBUG
struct PatternDebug {
    
    /// Generate a detailed report for pattern analysis (DEBUG only)
    static func debugReportPatterns(_ patterns: [CompositionalPattern],
                                  segmentsById: [UUID: AudioSegment],
                                  examplesById: [UUID: TrainingExample],
                                  cfg: PatternDiscoveryConfig) {
        
        guard !patterns.isEmpty else {
            print("🔍 Pattern Debug Report: No patterns to analyze")
            return
        }
        
        print("🔍 Pattern Debug Report - Analyzing \(patterns.count) patterns:")
        print("═══════════════════════════════════════════════════════════")
        
        let patternsToShow = min(5, patterns.count)
        let sortedPatterns = patterns.sorted { $0.confidence > $1.confidence }
        
        for (index, pattern) in sortedPatterns.prefix(patternsToShow).enumerated() {
            let cohesionScore = PatternValidator.cohesion(for: pattern, segmentsById: segmentsById)
            let consistencyScore = PatternValidator.meaningConsistency(for: pattern, examplesById: examplesById)
            
            print("[\(index + 1)] Pattern \(pattern.id.uuidString.prefix(8))...")
            print("    • Frequency: \(pattern.frequency)")
            print("    • Confidence: \(String(format: "%.3f", pattern.confidence))")
            print("    • Cohesion: \(String(format: "%.3f", cohesionScore))")
            print("    • Meaning Consistency: \(String(format: "%.3f", consistencyScore))")
            print("    • Average Position: \(String(format: "%.3f", pattern.averagePosition))")
            print("    • Associated Meanings: \(pattern.associatedMeanings.joined(separator: ", "))")
            
            let isValid = PatternValidator.isValid(pattern, 
                                                 segmentsById: segmentsById,
                                                 examplesById: examplesById,
                                                 cfg: cfg)
            print("    • Status: \(isValid ? "✅ Valid" : "❌ Invalid")")
            print("    ─────────────────────────────")
        }
        
        if patterns.count > patternsToShow {
            print("    ... and \(patterns.count - patternsToShow) more patterns")
        }
        
        // Summary statistics
        let validPatterns = patterns.filter { 
            PatternValidator.isValid($0, segmentsById: segmentsById, examplesById: examplesById, cfg: cfg) 
        }
        let validityRate = Float(validPatterns.count) / Float(patterns.count) * 100
        
        let avgFrequency = patterns.map { Float($0.frequency) }.reduce(0, +) / Float(patterns.count)
        let avgConfidence = patterns.map { $0.confidence }.reduce(0, +) / Float(patterns.count)
        
        print("\n📊 Summary Statistics:")
        print("    • Valid patterns: \(validPatterns.count)/\(patterns.count) (\(String(format: "%.1f", validityRate))%)")
        print("    • Average frequency: \(String(format: "%.1f", avgFrequency))")
        print("    • Average confidence: \(String(format: "%.3f", avgConfidence))")
        print("═══════════════════════════════════════════════════════════")
    }
}
#endif