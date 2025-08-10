//
//  ExamplesView.swift
//  RelayTive
//
//  Examples/Verification tab - manage and verify translations
//

import SwiftUI

struct ExamplesView: View {
    @EnvironmentObject var dataManager: DataManager
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
    
    var body: some View {
        HStack(spacing: 12) {
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
}

struct ExampleDetailView: View {
    @State private var editedUtterance: Utterance
    let onSave: (Utterance) -> Void
    @Environment(\.dismiss) private var dismiss
    
    init(utterance: Utterance, onSave: @escaping (Utterance) -> Void) {
        self._editedUtterance = State(initialValue: utterance)
        self.onSave = onSave
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Recording info section
                VStack(spacing: 12) {
                    Text("Original Recording")
                        .font(.headline)
                    
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
            .contentShape(Rectangle())
            .onTapGesture {
                // Dismiss keyboard when tapping outside
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
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
}


#Preview {
    ExamplesView()
        .environmentObject(DataManager())
}