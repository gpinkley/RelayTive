//
//  PhoneticDebugView.swift
//  RelayTive
//
//  SwiftUI debug interface for phonetic pipeline monitoring
//

import SwiftUI

// Local DTW result for debug view only
struct DTWResult {
    let distance: Float
    let insertions: Int
    let deletions: Int
    let substitutions: Int
}

struct PhoneticDebugView: View {
    @StateObject private var debugData = PhoneticDebugData()
    @State private var dtwInput1 = "U12 U7 U3"
    @State private var dtwInput2 = "U12 U8 U3"
    @State private var dtwResult: DTWResult?
    
    var phoneticEngine: PhoneticTranscriptionEngine?
    var classifier: NearestCentroidClassifier?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Live Metrics
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Live Metrics")
                            .font(.headline)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 8) {
                            MetricCard(
                                title: "Energy",
                                value: String(format: "%.1f dB", debugData.currentEnergy),
                                color: debugData.currentEnergy > -35 ? .green : .gray
                            )
                            
                            MetricCard(
                                title: "Spectral Flux",
                                value: String(format: "%.2f", debugData.currentFlux),
                                color: .orange
                            )
                            
                            MetricCard(
                                title: "Active Units",
                                value: "\(debugData.activeUnits)/160",
                                color: .green
                            )
                            
                            MetricCard(
                                title: "Confidence",
                                value: String(format: "%.2f", debugData.lastConfidence),
                                color: debugData.lastConfidence > 0.7 ? .green : .orange
                            )
                            
                            MetricCard(
                                title: "VAD State",
                                value: debugData.vadState.displayName,
                                color: debugData.vadState == .speech ? .green : .gray
                            )
                            
                            MetricCard(
                                title: "Cluster Purity",
                                value: String(format: "%.2f", debugData.clusterPurity),
                                color: .purple
                            )
                        }
                    }
                    
                    Divider()
                    
                    // Phonetic String
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Phonetic Transcription")
                            .font(.headline)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Unit String:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text(debugData.phoneticString.isEmpty ? "No units detected" : debugData.phoneticString)
                                .font(.system(.body, design: .monospaced))
                                .padding(8)
                                .background(Color(.systemGray6))
                                .cornerRadius(6)
                            
                            if let readable = debugData.readableSpelling {
                                Text("Readable:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 4)
                                
                                Text(readable)
                                    .font(.system(.body, design: .monospaced))
                                    .padding(8)
                                    .background(Color(.systemGray6))
                                    .cornerRadius(6)
                            }
                        }
                    }
                    
                    Divider()
                    
                    // DTW Testing
                    VStack(alignment: .leading, spacing: 8) {
                        Text("DTW Testing Tool")
                            .font(.headline)
                        
                        VStack(spacing: 8) {
                            HStack {
                                Text("String 1:")
                                    .font(.caption)
                                    .frame(width: 60, alignment: .leading)
                                
                                TextField("U12 U7 U3", text: $dtwInput1)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .font(.system(.caption, design: .monospaced))
                            }
                            
                            HStack {
                                Text("String 2:")
                                    .font(.caption)
                                    .frame(width: 60, alignment: .leading)
                                
                                TextField("U12 U8 U3", text: $dtwInput2)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .font(.system(.caption, design: .monospaced))
                            }
                            
                            Button("Calculate DTW") {
                                calculateDTW()
                            }
                            .buttonStyle(.bordered)
                            
                            if let result = dtwResult {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("DTW Result:")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    
                                    HStack {
                                        Text("Distance:")
                                            .font(.caption2)
                                        Spacer()
                                        Text(String(format: "%.2f", result.distance))
                                            .font(.system(.caption2, design: .monospaced))
                                    }
                                    
                                    HStack {
                                        Text("Operations:")
                                            .font(.caption2)
                                        Spacer()
                                        Text("I:\(result.insertions) D:\(result.deletions) S:\(result.substitutions)")
                                            .font(.system(.caption2, design: .monospaced))
                                    }
                                }
                                .padding(8)
                                .background(Color(.systemGray6))
                                .cornerRadius(6)
                            }
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Phonetic Debug")
        }
        .onAppear {
            debugData.startUpdating()
        }
        .onDisappear {
            debugData.stopUpdating()
        }
    }
    
    private func calculateDTW() {
        dtwResult = computeDTW(dtwInput1, dtwInput2)
    }
    
    private func computeDTW(_ a: String, _ b: String) -> DTWResult {
        let A = a.split(separator: " ").map(String.init)
        let B = b.split(separator: " ").map(String.init)
        let m = A.count, n = B.count
        guard m > 0 || n > 0 else { return DTWResult(distance: 0, insertions: 0, deletions: 0, substitutions: 0) }

        var dtw = Array(repeating: Array(repeating: Float.infinity, count: n + 1), count: m + 1)
        dtw[0][0] = 0
        
        for i in 1...m { dtw[i][0] = Float(i) }
        for j in 1...n { dtw[0][j] = Float(j) }

        for i in 1...m {
            for j in 1...n {
                let cost: Float = (A[i-1] == B[j-1]) ? 0 : 1
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
            }
        }

        // Backtrack to estimate ops
        var i = m, j = n
        var ins = 0, del = 0, sub = 0
        while i > 0 || j > 0 {
            if i == 0 { ins += 1; j -= 1; continue }
            if j == 0 { del += 1; i -= 1; continue }
            let up = dtw[i-1][j], left = dtw[i][j-1], diag = dtw[i-1][j-1]
            if diag <= up && diag <= left {
                if A[i-1] != B[j-1] { sub += 1 }
                i -= 1; j -= 1
            } else if up < left {
                del += 1; i -= 1
            } else {
                ins += 1; j -= 1
            }
        }

        return DTWResult(distance: dtw[m][n], insertions: ins, deletions: del, substitutions: sub)
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.medium)
                .foregroundColor(color)
        }
        .padding(8)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

class PhoneticDebugData: ObservableObject {
    @Published var currentEnergy: Float = -60.0
    @Published var currentFlux: Float = 0.0
    @Published var activeUnits: Int = 0
    @Published var lastConfidence: Float = 0.0
    @Published var vadState: VADProcessor.VADState = .silence
    @Published var clusterPurity: Float = 0.0
    @Published var phoneticString: String = ""
    @Published var readableSpelling: String? = nil
    
    private var updateTimer: Timer?
    
    func startUpdating() {
        updateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            self.updateMetrics()
        }
    }
    
    func stopUpdating() {
        updateTimer?.invalidate()
    }
    
    private func updateMetrics() {
        currentEnergy = -45.0 + Float.random(in: -10...10)
        currentFlux = max(0, Float.random(in: 0...2.0))
        
        if Int.random(in: 0...10) == 0 {
            activeUnits = min(160, max(0, activeUnits + Int.random(in: -5...5)))
        }
    }
    
    func updateWithTranscription(_ transcription: PhoneticTranscription) {
        DispatchQueue.main.async {
            self.phoneticString = transcription.unitString
            self.readableSpelling = transcription.readableSpelling
        }
    }
    
    func updateWithClassification(_ result: ClassificationResult) {
        DispatchQueue.main.async {
            self.lastConfidence = result.confidence
        }
    }
    
    func updateWithVAD(_ result: VADResult) {
        DispatchQueue.main.async {
            self.currentEnergy = result.energy
            self.currentFlux = result.spectralFlux
            self.vadState = result.state
        }
    }
}

extension VADProcessor.VADState {
    var displayName: String {
        switch self {
        case .silence: return "Silence"
        case .speechStart: return "Starting"
        case .speech: return "Speech"
        case .speechEnd: return "Ending"
        case .hangover: return "Hangover"
        }
    }
}
