//
//  AudioManager.swift
//  RelayTive
//
//  Manages audio recording and playback functionality
//

import Foundation
import AVFoundation
import SwiftUI

@MainActor
class AudioManager: ObservableObject {
    @Published var isRecording = false
    @Published var isPlaying = false
    
    private var audioRecorder: AVAudioRecorder?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegate?
    private var recordingSession: AVAudioSession = AVAudioSession.sharedInstance()
    
    // Temporary recording storage
    private var currentRecordingURL: URL?
    
    init() {
        setupAudioSession()
    }
    
    // MARK: - Audio Session Setup
    
    private func setupAudioSession() {
        do {
            // Configure for both recording and playback with proper routing
            try recordingSession.setCategory(.playAndRecord, 
                                           mode: .default, 
                                           options: [.defaultToSpeaker, .allowBluetooth])
            try recordingSession.setActive(true)
            print("Audio session configured for record/playback with speaker output")
        } catch {
            print("Failed to set up recording session: \(error)")
        }
    }
    
    // MARK: - Recording
    
    func startRecording() {
        // Request microphone permission
        requestMicrophonePermission { [weak self] granted in
            if granted {
                Task { @MainActor in
                    self?.beginRecording()
                }
            } else {
                print("Microphone permission denied")
            }
        }
    }
    
    private func beginRecording() {
        // Create temporary file URL
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        currentRecordingURL = documentsPath.appendingPathComponent("temp_recording_\(UUID().uuidString).wav")
        
        // Configure recording settings
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 44100,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        
        do {
            audioRecorder = try AVAudioRecorder(url: currentRecordingURL!, settings: settings)
            audioRecorder?.prepareToRecord()
            audioRecorder?.record()
            isRecording = true
            print("Recording started")
        } catch {
            print("Failed to start recording: \(error)")
        }
    }
    
    func stopRecording() -> Data? {
        guard isRecording, let recorder = audioRecorder else { return nil }
        
        recorder.stop()
        isRecording = false
        
        // Convert recorded file to Data
        if let url = currentRecordingURL {
            do {
                let audioData = try Data(contentsOf: url)
                print("Recording stopped, data size: \(audioData.count) bytes")
                
                // Clean up temporary file
                try? FileManager.default.removeItem(at: url)
                currentRecordingURL = nil
                
                return audioData
            } catch {
                print("Failed to convert recording to data: \(error)")
            }
        }
        
        return nil
    }
    
    // MARK: - Playback
    
    func playAudio(_ audioData: Data) {
        // Create temporary file for playback
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("playback_\(UUID().uuidString).wav")
        
        do {
            try audioData.write(to: tempURL)
            audioPlayer = try AVAudioPlayer(contentsOf: tempURL)
            
            // Configure for proper playback volume
            audioPlayer?.volume = 1.0
            audioPlayer?.prepareToPlay()
            
            // Ensure audio session is configured for playback
            try recordingSession.setCategory(.playAndRecord, 
                                           mode: .default, 
                                           options: [.defaultToSpeaker, .allowBluetooth])
            try recordingSession.overrideOutputAudioPort(.speaker)
            
            // Keep a strong reference to the delegate
            audioPlayerDelegate = AudioPlayerDelegate { [weak self] in
                Task { @MainActor in
                    self?.isPlaying = false
                    // Reset audio session after playback
                    try? self?.recordingSession.overrideOutputAudioPort(.none)
                    // Clean up temp file
                    try? FileManager.default.removeItem(at: tempURL)
                }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            
            audioPlayer?.play()
            isPlaying = true
            print("Playing audio at full volume, duration: \(audioPlayer?.duration ?? 0) seconds")
        } catch {
            print("Failed to play audio: \(error)")
        }
    }
    
    func stopPlayback() {
        audioPlayer?.stop()
        isPlaying = false
    }
    
    // MARK: - Permissions
    
    private func requestMicrophonePermission(completion: @escaping (Bool) -> Void) {
        if #available(iOS 17.0, *) {
            // Use modern AVAudioApplication API for iOS 17+
            switch AVAudioApplication.shared.recordPermission {
            case .granted:
                completion(true)
            case .denied:
                completion(false)
            case .undetermined:
                AVAudioApplication.requestRecordPermission { granted in
                    DispatchQueue.main.async {
                        completion(granted)
                    }
                }
            @unknown default:
                completion(false)
            }
        } else {
            // Fallback to deprecated API for iOS 16 and earlier
            switch recordingSession.recordPermission {
            case .granted:
                completion(true)
            case .denied:
                completion(false)
            case .undetermined:
                recordingSession.requestRecordPermission { granted in
                    DispatchQueue.main.async {
                        completion(granted)
                    }
                }
            @unknown default:
                completion(false)
            }
        }
    }
    
    // MARK: - Cleanup
    
    deinit {
        audioRecorder?.stop()
        audioPlayer?.stop()
    }
}

// MARK: - Audio Player Delegate Helper

private class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate {
    private let onFinish: () -> Void
    
    init(onFinish: @escaping () -> Void) {
        self.onFinish = onFinish
    }
    
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish()
    }
}