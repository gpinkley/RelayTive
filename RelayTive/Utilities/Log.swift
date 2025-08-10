//
//  Log.swift
//  RelayTive
//
//  Logging utility with verbosity control
//

import Foundation

struct Log {
    #if DEBUG
    static var isVerbose = false  // Can be enabled for detailed debugging
    #else
    static var isVerbose = false  // Always false in release
    #endif
    
    /// Print only if verbose logging is enabled
    static func verbose(_ message: String) {
        if isVerbose {
            print(message)
        }
    }
    
    /// Always print (for important messages)
    static func info(_ message: String) {
        print(message)
    }
    
    /// Print warnings
    static func warning(_ message: String) {
        print("⚠️ \(message)")
    }
    
    /// Print errors
    static func error(_ message: String) {
        print("❌ \(message)")
    }
    
    /// Print success messages
    static func success(_ message: String) {
        print("✅ \(message)")
    }
}