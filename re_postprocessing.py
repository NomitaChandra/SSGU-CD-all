# RE Post-processing Script
# Run after RE output to put the predicted outputs into a csv format that can be used to enter information to databases

import pandas as pd
import csv
import re
from collections import defaultdict

# Load MeSH lookup table to map MeSH IDs to their official names
def load_mesh_lookup(file_path):
    mesh_lookup_df = pd.read_csv(file_path)
    # Create a dictionary with MeSH ID as key and the official name as value
    mesh_lookup = {}
    for _, row in mesh_lookup_df.iterrows():
        mesh_id = row['MeSH ID']
        official_name = row['Names/Entry Terms'].split('|')[0].strip()  # Take the first term as the official name
        mesh_lookup[mesh_id] = official_name
    return mesh_lookup

# Function to parse RE output from txt file
def parse_re_output(file_path):
    papers = defaultdict(lambda: {
        "title": "",
        "abstract": "",
        "entities": [],
        "relationships": []
    })
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if it's a title or abstract line
            if "|t|" in line or "|a|" in line:
                paper_id, section, text = line.split("|", 2)
                if section == "t":
                    papers[paper_id]["title"] = text
                elif section == "a":
                    papers[paper_id]["abstract"] = text
            
            # Check if it's an entity line
            elif re.match(r"\d+\t\d+\t\d+\t", line):
                parts = line.split("\t")
                if len(parts) == 6:
                    paper_id, start, end, entity, entity_type, mesh_id = parts
                    papers[paper_id]["entities"].append({
                        "start": int(start),
                        "end": int(end),
                        "name": entity,
                        "type": entity_type,
                        "mesh_id": mesh_id
                    })
            
            # Check if it's a relationship prediction line
            elif "\tCID\t" in line:
                parts = line.split("\t")
                if len(parts) == 5:
                    paper_id = parts[0]
                    mesh_id_1 = parts[2]
                    mesh_id_2 = parts[3]
                    papers[paper_id]["relationships"].append({
                        "mesh_id_1": mesh_id_1,
                        "mesh_id_2": mesh_id_2
                    })

    return papers


# Function to write entity information to a CSV file
def write_entities_to_csv(papers, mesh_lookup, output_path):
    with open(output_path, mode='w', newline='') as csv_file:
        fieldnames = ["paper_id", "title", "abstract", "start", "end", "entity_name", "entity_type", "mesh_id", "normalized_name"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for paper_id, paper_data in papers.items():
            title = paper_data["title"]
            abstract = paper_data["abstract"]
            for entity in paper_data["entities"]:
                normalized_name = mesh_lookup.get(entity["mesh_id"], "Unknown")
                writer.writerow({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "start": entity["start"],
                    "end": entity["end"],
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "mesh_id": entity["mesh_id"],
                    "normalized_name": normalized_name
                })

# Function to write relationship information to a separate CSV file
def write_relationships_to_csv(papers, mesh_lookup, output_path):
    with open(output_path, mode='w', newline='') as csv_file:
        fieldnames = ["paper_id", "title", "abstract", "mesh_id_1", "mesh_id_2", "normalized_name_1", "normalized_name_2"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for paper_id, paper_data in papers.items():
            title = paper_data["title"]
            abstract = paper_data["abstract"]
            for relationship in paper_data["relationships"]:
                normalized_name_1 = mesh_lookup.get(relationship["mesh_id_1"], "Unknown")
                normalized_name_2 = mesh_lookup.get(relationship["mesh_id_2"], "Unknown")
                writer.writerow({
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "mesh_id_1": relationship["mesh_id_1"],
                    "mesh_id_2": relationship["mesh_id_2"],
                    "normalized_name_1": normalized_name_1,
                    "normalized_name_2": normalized_name_2
                })

# Main function
def main(input_file, mesh_lookup_file, entity_output_file, relationship_output_file):
    # Load the MeSH lookup table
    mesh_lookup = load_mesh_lookup(mesh_lookup_file)
    
    # Parse the RE output
    papers = parse_re_output(input_file)
    
    # Write entity information to a CSV file with official names
    write_entities_to_csv(papers, mesh_lookup, entity_output_file)
    
    # Write relationship information to a separate CSV file with official names
    write_relationships_to_csv(papers, mesh_lookup, relationship_output_file)

# Run
if __name__ == "__main__":
    # When running .py file, below can be commented out and instead be defined outside by .py call 
    re_postprocessing_input_file = "/Users/kavithakamarthy/Downloads/SSGU-CD-all/result/cdr/cdr_results.pubtator"  # Replace with your input RE output file
    mesh_lookup_file = "mesh_lookup_table_with_dsstox.csv"  
    entity_output_file = "re_output_postprocessed_entities.csv"  # Output CSV file for entities
    relationship_output_file = "re_output_postprocessed_relationships.csv"  # Output CSV file for relationships
    
    main(re_postprocessing_input_file, mesh_lookup_file, entity_output_file, relationship_output_file)
