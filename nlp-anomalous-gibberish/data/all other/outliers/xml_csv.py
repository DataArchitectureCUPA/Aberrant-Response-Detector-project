import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob

folder_path = "C:\\Users\\mpagoni\\OneDrive - Cambridge\\Desktop\\Aberrant Response detector project\\Bitbucket\\nlp-anomalous-gibberish\\data\\outliers\\0102_2001_12"  # Change this if your XMLs are in a different folder
output_csv = "output4.csv"

# List to store the data
data = []

# Process each XML file
xml_files = glob.glob(os.path.join(folder_path, "**/*.xml"), recursive=True)
print(f"Found {len(xml_files)} XML files")

for xml_file in xml_files:
    try:
        print(f"Processing: {xml_file}")
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get sortkey from head element
        head = root.find('head')
        sortkey = head.get('sortkey') if head is not None else ''
        
        # Get all text content, joining paragraphs with spaces
        text_elem = root.find('.//text')
        if text_elem is not None:
            # Get all paragraphs and join them with spaces
            paragraphs = text_elem.findall('p')
            text_content = ' '.join(p.text for p in paragraphs if p.text)
        else:
            text_content = ''
        
        data.append({
            'sortkey': sortkey,
            'text': text_content,
            'filename': os.path.basename(xml_file)
        })
        
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

# Create and save CSV
if data:
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Created CSV file with {len(data)} entries")
    print(f"Output saved to: {os.path.abspath(output_csv)}")
else:
    print("No data was processed")
    print("\nFiles in data directory:")
    print(os.listdir(folder_path))