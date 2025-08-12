//
//  OnlineKMeans.swift
//  RelayTive
//
//  Online K-Means clustering in cosine space with EMA updates
//

import Foundation
import Accelerate

/// Online K-Means clustering optimized for cosine similarity
class OnlineKMeans {
    
    // MARK: - Configuration
    private let k: Int
    private let dim: Int
    private let baseLearningRate: Float
    private let minLearningRate: Float
    private let decayFactor: Float
    
    // MARK: - State
    private var centroids: [[Float]]
    private var clusterCounts: [Int]
    private var totalObservations: Int = 0
    
    // Thread safety
    private let queue = DispatchQueue(label: "com.relayTive.onlineKMeans", qos: .utility)
    
    // MARK: - Initialization
    
    init(k: Int = 160, dim: Int, baseLearningRate: Float = 0.1, minLearningRate: Float = 0.001, decayFactor: Float = 0.95) {
        self.k = k
        self.dim = dim
        self.baseLearningRate = baseLearningRate
        self.minLearningRate = minLearningRate
        self.decayFactor = decayFactor
        
        // Initialize centroids randomly on unit sphere
        self.centroids = (0..<k).map { _ in
            Self.randomUnitVector(dim: dim)
        }
        self.clusterCounts = Array(repeating: 0, count: k)
        
        print("📊 OnlineKMeans initialized: k=\(k), dim=\(dim)")
    }
    
    // MARK: - Public Interface
    
    /// Observe a new vector and return assigned cluster ID
    func observe(_ vector: [Float]) -> Int {
        guard vector.count == dim else {
            print("❌ Vector dimension mismatch: expected \(dim), got \(vector.count)")
            return 0
        }
        
        return queue.sync {
            // L2 normalize input vector
            let normalizedVector = l2Normalize(vector)
            
            // Find closest centroid using cosine similarity (dot product since normalized)
            let clusterId = findClosestCentroid(normalizedVector)
            
            // Update centroid with EMA
            updateCentroid(clusterId, with: normalizedVector)
            
            // Update statistics
            clusterCounts[clusterId] += 1
            totalObservations += 1
            
            return clusterId
        }
    }
    
    /// Get current centroids (L2 normalized)
    func getCentroids() -> [[Float]] {
        return queue.sync {
            return centroids.map { l2Normalize($0) }
        }
    }
    
    /// Get cluster sizes
    func getClusterSizes() -> [Int] {
        return queue.sync {
            return clusterCounts
        }
    }
    
    /// Reset clustering state
    func reset() {
        queue.sync {
            centroids = (0..<k).map { _ in
                Self.randomUnitVector(dim: dim)
            }
            clusterCounts = Array(repeating: 0, count: k)
            totalObservations = 0
        }
        print("📊 OnlineKMeans reset")
    }
    
    /// Get statistics
    func getStatistics() -> ClusteringStatistics {
        return queue.sync {
            let activeClusters = clusterCounts.filter { $0 > 0 }.count
            let avgClusterSize = totalObservations > 0 ? Float(totalObservations) / Float(activeClusters) : 0
            let maxClusterSize = clusterCounts.max() ?? 0
            let minClusterSize = clusterCounts.filter { $0 > 0 }.min() ?? 0
            
            // Calculate cluster purity (how well separated clusters are)
            let purity = calculateClusterPurity()
            
            return ClusteringStatistics(
                totalObservations: totalObservations,
                activeClusters: activeClusters,
                averageClusterSize: avgClusterSize,
                maxClusterSize: maxClusterSize,
                minClusterSize: minClusterSize,
                clusterPurity: purity
            )
        }
    }
    
    /// Force centroid normalization (periodic maintenance)
    func normalizeCentroids() {
        queue.sync {
            for i in 0..<k {
                centroids[i] = l2Normalize(centroids[i])
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func findClosestCentroid(_ vector: [Float]) -> Int {
        var bestCluster = 0
        var bestSimilarity: Float = -1.0
        
        for i in 0..<k {
            let similarity = dotProduct(vector, centroids[i])
            if similarity > bestSimilarity {
                bestSimilarity = similarity
                bestCluster = i
            }
        }
        
        return bestCluster
    }
    
    private func updateCentroid(_ clusterId: Int, with vector: [Float]) {
        // Calculate adaptive learning rate based on cluster size
        let clusterSize = clusterCounts[clusterId]
        let adaptiveLR = calculateLearningRate(clusterSize: clusterSize)
        
        // EMA update: centroid = (1 - lr) * centroid + lr * vector
        let oneMinusLR = 1.0 - adaptiveLR
        
        for i in 0..<dim {
            centroids[clusterId][i] = oneMinusLR * centroids[clusterId][i] + adaptiveLR * vector[i]
        }
        
        // L2 normalize updated centroid to maintain unit sphere constraint
        centroids[clusterId] = l2Normalize(centroids[clusterId])
    }
    
    private func calculateLearningRate(clusterSize: Int) -> Float {
        // Adaptive learning rate: starts high, decreases with cluster size
        let sizeFactor = 1.0 / (1.0 + Float(clusterSize) * 0.01)
        let lr = baseLearningRate * sizeFactor * pow(decayFactor, Float(totalObservations) / 1000.0)
        return max(lr, minLearningRate)
    }
    
    private func calculateClusterPurity() -> Float {
        guard totalObservations > 0 else { return 0.0 }
        
        // Calculate average inter-cluster similarity  
        var interClusterSim: Float = 0.0
        var comparisons = 0
        
        // Sample pairwise similarities between centroids
        for i in 0..<k {
            for j in (i+1)..<k {
                let similarity = dotProduct(centroids[i], centroids[j])
                interClusterSim += similarity
                comparisons += 1
            }
        }
        
        if comparisons > 0 {
            interClusterSim /= Float(comparisons)
        }
        
        // Purity = how well separated clusters are (lower inter-cluster similarity = higher purity)
        return max(0.0, 1.0 - interClusterSim)
    }
    
    // MARK: - Vector Operations (using Accelerate)
    
    private func l2Normalize(_ vector: [Float]) -> [Float] {
        var result = vector
        var norm: Float = 0.0
        
        // Calculate squared norm
        vDSP_svesq(vector, 1, &norm, vDSP_Length(vector.count))
        norm = sqrt(norm)
        
        // Avoid division by zero
        if norm > 1e-10 {
            vDSP_vsdiv(vector, 1, &norm, &result, 1, vDSP_Length(vector.count))
        }
        
        return result
    }
    
    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0.0 }
        
        var result: Float = 0.0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }
    
    private static func randomUnitVector(dim: Int) -> [Float] {
        // Generate random Gaussian vector and normalize
        var vector = (0..<dim).map { _ in Float.random(in: -1...1) }
        
        // L2 normalize
        var norm: Float = 0.0
        vDSP_svesq(vector, 1, &norm, vDSP_Length(dim))
        norm = sqrt(norm)
        
        if norm > 1e-10 {
            vDSP_vsdiv(vector, 1, &norm, &vector, 1, vDSP_Length(dim))
        }
        
        return vector
    }
}

// MARK: - Support Structures

struct ClusteringStatistics {
    let totalObservations: Int
    let activeClusters: Int
    let averageClusterSize: Float
    let maxClusterSize: Int
    let minClusterSize: Int
    let clusterPurity: Float
    
    var utilizationRate: Float {
        guard activeClusters > 0 else { return 0.0 }
        return Float(activeClusters) / 160.0  // Assuming k=160
    }
    
    var isWellDistributed: Bool {
        // Check if cluster sizes are reasonably balanced
        guard maxClusterSize > 0 else { return false }
        let sizeRatio = Float(minClusterSize) / Float(maxClusterSize)
        return sizeRatio > 0.1 && clusterPurity > 0.3
    }
}

// MARK: - Codebook Management Extension

extension OnlineKMeans {
    
    /// Save codebook state for persistence
    func saveCodebook() -> CodebookData {
        return queue.sync {
            return CodebookData(
                centroids: getCentroids(),
                clusterCounts: clusterCounts,
                totalObservations: totalObservations,
                k: k,
                dim: dim
            )
        }
    }
    
    /// Load codebook state from persistence
    func loadCodebook(_ data: CodebookData) {
        guard data.k == k && data.dim == dim else {
            print("❌ Codebook dimension mismatch: expected k=\(k), dim=\(dim), got k=\(data.k), dim=\(data.dim)")
            return
        }
        
        queue.sync {
            self.centroids = data.centroids
            self.clusterCounts = data.clusterCounts
            self.totalObservations = data.totalObservations
        }
        
        print("📊 OnlineKMeans codebook loaded: \(totalObservations) observations, \(getStatistics().activeClusters) active clusters")
    }
}

struct CodebookData: Codable {
    let centroids: [[Float]]
    let clusterCounts: [Int]
    let totalObservations: Int
    let k: Int
    let dim: Int
    let timestamp: Date
    
    init(centroids: [[Float]], clusterCounts: [Int], totalObservations: Int, k: Int, dim: Int) {
        self.centroids = centroids
        self.clusterCounts = clusterCounts
        self.totalObservations = totalObservations
        self.k = k
        self.dim = dim
        self.timestamp = Date()
    }
}