import csv
import json
from pathlib import Path

# Define file paths
input_csv = Path(r"D:\RWKV-Translator\Chinese to english dataset.csv")
output_jsonl = Path(r"D:\RWKV-Translator\dataset.json")  

# Open input CSV and output file
with open(input_csv, 'r', encoding='utf-8') as csvfile, \
     open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    
    reader = csv.reader(csvfile)
    
    # Skip header row
    next(reader)
    
    # Process rows 2 to 2617 (starting from row 2 in 1‑based counting)
    for row_num, row in enumerate(reader, start=2):
        if row_num > 2617:
            break
        
        # Ensure row has at least two columns
        if len(row) >= 2:
            english = row[0].strip()
            chinese = row[1].strip()
            
            # Build the text field as required
            text_content = f"User: {english}\n\nAssistant: {chinese}"
            
            # Create a JSON object and write it as a line
            record = {"text": text_content}
            jsonlfile.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Successfully converted rows 2–2617 to {output_jsonl}")