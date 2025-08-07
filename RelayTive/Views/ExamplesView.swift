//
//  ExamplesView.swift
//  RelayTive
//
//  Examples/Verification tab - manage and verify translations
//

import SwiftUI

struct ExamplesView: View {
    @EnvironmentObject var dataManager: DataManager
    @EnvironmentObject var audioManager: AudioManager
    @State private var searchText = ""
    @State private var filterVerified: FilterOption = .all
    @State private var selectedUtterance: Utterance?
    
    enum FilterOption: String, CaseIterable {
        case all = "All"
        case verified = "Verified"
        case unverified = "Unverified"
    }
    
    var filteredUtterances: [Utterance] {
        let filtered = dataManager.allUtterances.filter { utterance in
            let matchesSearch = searchText.isEmpty || utterance.translation.localizedCaseInsensitiveContains(searchText)
            let matchesFilter = filterVerified == .all ||
                               (filterVerified == .verified && utterance.isVerified) ||
                               (filterVerified == .unverified && !utterance.isVerified)
            return matchesSearch && matchesFilter
        }
        return filtered.sorted { $0.timestamp > $1.timestamp }
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Search and filter
                VStack(spacing: 16) {
                    // Search bar
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.secondary)
                        
                        TextField("Search translations...", text: $searchText)
                            .textFieldStyle(.plain)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Filter options
                    Picker("Filter", selection: $filterVerified) {
                        ForEach(FilterOption.allCases, id: \.self) { option in
                            Text(option.rawValue).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                .padding()
                
                // Results count
                HStack {
                    Text("\(filteredUtterances.count) examples")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                
                // Examples list
                List {
                    ForEach(filteredUtterances) { utterance in
                        ExampleRow(utterance: utterance) {
                            selectedUtterance = utterance
                        }
                    }
                    .onDelete(perform: deleteUtterances)
                }
                .listStyle(.plain)
            }
            .navigationTitle("Examples")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    EditButton()
                }
            }
        }
        .sheet(item: $selectedUtterance) { utterance in
            ExampleDetailView(utterance: utterance) { updatedUtterance in
                dataManager.updateUtterance(updatedUtterance)
                selectedUtterance = nil
            }
        }
    }
    
    private func deleteUtterances(at offsets: IndexSet) {
        for index in offsets {
            let utterance = filteredUtterances[index]
            dataManager.removeUtterance(utterance)
        }
    }
}

struct ExampleRow: View {
    let utterance: Utterance
    let onTap: () -> Void
    @EnvironmentObject var audioManager: AudioManager
    
    var body: some View {
        HStack(spacing: 12) {
            // Play button
            Button(action: playAudio) {
                Image(systemName: "play.circle.fill")
                    .font(.title2)
                    .foregroundColor(.blue)
            }
            .buttonStyle(PlainButtonStyle())
            
            // Content
            VStack(alignment: .leading, spacing: 4) {
                Text(utterance.translation)
                    .font(.body)
                    .foregroundColor(.primary)
                
                HStack {
                    Text(utterance.timestamp, style: .date)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    if utterance.isVerified {
                        Image(systemName: "checkmark.seal.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                    }
                }
            }
            
            Spacer()
            
            // Edit button
            Button(action: onTap) {
                Image(systemName: "pencil.circle")
                    .font(.title3)
                    .foregroundColor(.gray)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            onTap()
        }
    }
    
    private func playAudio() {
        // TODO: Play the original audio recording
        audioManager.playAudio(utterance.originalAudio)
    }
}

struct ExampleDetailView: View {
    @State private var editedUtterance: Utterance
    let onSave: (Utterance) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var words: [WordSegment] = []
    @State private var editingMode = false
    
    init(utterance: Utterance, onSave: @escaping (Utterance) -> Void) {
        self._editedUtterance = State(initialValue: utterance)
        self.onSave = onSave
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Audio playback section
                VStack(spacing: 12) {
                    Text("Original Recording")
                        .font(.headline)
                    
                    Button(action: playOriginalAudio) {
                        HStack {
                            Image(systemName: "play.circle.fill")
                            Text("Play Recording")
                        }
                        .font(.title3)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Text("Recorded on \(editedUtterance.timestamp, style: .date)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Divider()
                
                // Translation editing section
                VStack(alignment: .leading, spacing: 12) {
                    Text("Translation")
                        .font(.headline)
                    
                    TextEditor(text: $editedUtterance.translation)
                        .frame(minHeight: 100)
                        .padding(8)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    
                    Text("Edit to be as literal as possible")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Word-level breakdown section
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Word Breakdown")
                            .font(.headline)
                        
                        Spacer()
                        
                        Button(editingMode ? "Done" : "Edit Words") {
                            if editingMode {
                                // Save word changes back to translation
                                editedUtterance.translation = words.map(\.text).joined(separator: " ")
                            } else {
                                // Split current translation into words
                                words = editedUtterance.translation
                                    .split(separator: " ")
                                    .map { WordSegment(text: String($0)) }
                            }
                            editingMode.toggle()
                        }
                        .font(.caption)
                        .buttonStyle(.bordered)
                    }
                    
                    if editingMode {
                        WordEditingView(words: $words)
                    } else {
                        Text("Tap 'Edit Words' to break down into individual words for fine-grained editing")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .italic()
                    }
                }
                
                Spacer()
                
                // Verification toggle
                HStack {
                    Toggle("Mark as Verified", isOn: $editedUtterance.isVerified)
                        .font(.headline)
                    
                    Spacer()
                    
                    if editedUtterance.isVerified {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundColor(.green)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
            }
            .padding()
            .navigationTitle("Edit Example")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        onSave(editedUtterance)
                    }
                    .fontWeight(.semibold)
                }
            }
        }
    }
    
    private func playOriginalAudio() {
        // TODO: Play the original audio
    }
}

struct WordSegment: Identifiable {
    let id = UUID()
    var text: String
}

struct WordEditingView: View {
    @Binding var words: [WordSegment]
    @State private var selectedWordIndex: Int?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tap words to edit, drag to reorder")
                .font(.caption)
                .foregroundColor(.secondary)
            
            LazyVGrid(columns: [
                GridItem(.adaptive(minimum: 80), spacing: 8)
            ], spacing: 8) {
                ForEach(Array(words.enumerated()), id: \.element.id) { index, word in
                    WordChip(
                        word: word.text,
                        isSelected: selectedWordIndex == index
                    ) {
                        selectedWordIndex = index
                    }
                }
            }
            
            if let selectedIndex = selectedWordIndex {
                HStack {
                    TextField("Edit word", text: $words[selectedIndex].text)
                        .textFieldStyle(.roundedBorder)
                    
                    Button("Delete") {
                        words.remove(at: selectedIndex)
                        selectedWordIndex = nil
                    }
                    .font(.caption)
                    .foregroundColor(.red)
                }
            }
            
            Button("Add Word") {
                words.append(WordSegment(text: "new"))
                selectedWordIndex = words.count - 1
            }
            .font(.caption)
            .buttonStyle(.bordered)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

struct WordChip: View {
    let word: String
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Text(word)
            .font(.body)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(isSelected ? Color.blue : Color(.systemGray5))
            .foregroundColor(isSelected ? .white : .primary)
            .cornerRadius(20)
            .onTapGesture {
                onTap()
            }
    }
}

#Preview {
    ExamplesView()
        .environmentObject(DataManager())
        .environmentObject(AudioManager())
}