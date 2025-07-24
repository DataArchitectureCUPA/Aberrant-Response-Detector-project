import os
import xml.etree.ElementTree as ET
import pandas as pd
import glob

def extract_xml_data(xml_directory, output_file):
    """
    Extract sortkey from head element and text content from XML files
    and save to CSV.
    
    Parameters:
    xml_directory (str): Path to directory containing XML files
    output_file (str): Path for output CSV file
    """
    # List to store all extracted data
    all_records = []
    
    # Get all XML files in directory and subdirectories
    xml_files = glob.glob(os.path.join(xml_directory, "**/*.xml"), recursive=True)
    
    for xml_file in xml_files:
        try:
            # Parse XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find head element and get sortkey
            head = root.find('.//head')
            sortkey = head.get('sortkey') if head is not None else ''
            
            # Find text element and get content
            text_elem = root.find('.//text')
            text_content = text_elem.text if text_elem is not None else ''
            
            # Add to records
            all_records.append({
                'sortkey': sortkey,
                'text': text_content,
                'source_file': os.path.basename(xml_file)  # Optional: keep track of source
            })
            
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
    
    # Convert to DataFrame and save as CSV
    if all_records:
        df = pd.DataFrame(all_records)
        # If you don't want the source_file column, uncomment the next line
        # df = df[['sortkey', 'text']]  
        df.to_csv(output_file, index=False)
        print(f"Successfully created CSV file: {output_file}")
        print(f"Processed {len(all_records)} XML files")
    else:
        print("No data was processed. Check your XML files and structure.")

# Example usage
if __name__ == "__main__":
    xml_directory = "0101_2001_6"  # Replace with your directory path
    output_file = "output.csv"
    extract_xml_data(xml_directory, output_file)