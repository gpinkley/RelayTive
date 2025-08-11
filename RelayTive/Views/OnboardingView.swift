//
//  OnboardingView.swift
//  RelayTive
//
//  Permission onboarding screen for microphone and speech recognition
//

import SwiftUI
import Speech
import AVFoundation

struct OnboardingView: View {
    @State private var isRequestingPermissions = false
    @State private var permissionError: String?
    
    let onComplete: () -> Void
    
    var body: some View {
        VStack(spacing: 40) {
            Spacer()
            
            // App icon or logo area
            VStack(spacing: 16) {
                Image(systemName: "mic.badge.plus")
                    .font(.system(size: 80))
                    .foregroundColor(.blue)
                
                Text("Welcome to RelayTive")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
            }
            
            // Permission explanation
            VStack(spacing: 24) {
                VStack(spacing: 12) {
                    HStack(spacing: 16) {
                        Image(systemName: "mic.fill")
                            .font(.title2)
                            .foregroundColor(.blue)
                            .frame(width: 30)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Microphone Access")
                                .font(.headline)
                            Text("To record your unique vocalizations")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                    }
                    
                    HStack(spacing: 16) {
                        Image(systemName: "waveform")
                            .font(.title2)
                            .foregroundColor(.blue)
                            .frame(width: 30)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Speech Recognition")
                                .font(.headline)
                            Text("To understand caregiver explanations")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(16)
                
                VStack(spacing: 8) {
                    Text("RelayTive helps translate atypical speech patterns into clear communication.")
                        .font(.body)
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                    
                    Text("All processing happens on your device. Nothing is shared.")
                        .font(.caption)
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                }
            }
            
            // Error message if any
            if let error = permissionError {
                Text(error)
                    .font(.body)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .padding()
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
            }
            
            Spacer()
            
            // Continue button
            Button(action: requestPermissions) {
                HStack {
                    if isRequestingPermissions {
                        ProgressView()
                            .scaleEffect(0.8)
                            .tint(.white)
                    } else {
                        Text("Continue")
                            .fontWeight(.semibold)
                    }
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .cornerRadius(12)
            }
            .disabled(isRequestingPermissions)
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 20)
    }
    
    // MARK: - Permission Request Flow
    
    private func requestPermissions() {
        isRequestingPermissions = true
        permissionError = nil
        
        Task {
            do {
                // Request both permissions
                let micGranted = await requestMicrophonePermission()
                let speechGranted = await requestSpeechPermission()
                
                await MainActor.run {
                    if micGranted && speechGranted {
                        // Success - complete onboarding
                        onComplete()
                    } else {
                        // Handle denied permissions
                        var errors: [String] = []
                        if !micGranted {
                            errors.append("Microphone access is required to record your speech")
                        }
                        if !speechGranted {
                            errors.append("Speech recognition is required to understand explanations")
                        }
                        
                        permissionError = errors.joined(separator: "\n")
                        isRequestingPermissions = false
                    }
                }
                
            } catch {
                await MainActor.run {
                    permissionError = "Permission request failed: \(error.localizedDescription)"
                    isRequestingPermissions = false
                }
            }
        }
    }
    
    private func requestMicrophonePermission() async -> Bool {
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
    
    private func requestSpeechPermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }
}

#Preview {
    OnboardingView {
        print("Onboarding completed")
    }
}