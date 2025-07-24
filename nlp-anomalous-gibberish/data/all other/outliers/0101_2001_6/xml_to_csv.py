import xml.etree.ElementTree as ET
import csv

# Replace with the path to your XML file
xml_file = 'C:\\Users\\mpagoni\\OneDrive - Cambridge\\Desktop\\Aberrant Response detector project\\Bitbucket\\nlp-anomalous-gibberish\\data\\outliers\\0101_2001_6\\1_doc3162.xml'
csv_file = 'output.csv'

# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Open CSV file for writing
with open(csv_file, 'w', newline='', encoding='utf-8') as csvf:
    writer = csv.writer(csvf)

    # Extract the headers (assuming the XML structure has a repetitive pattern)
    headers = []
    for child in root[0]:
        headers.append(child.tag)
    writer.writerow(headers)  # Write header to CSV file

    # Extract and write the data
    for elem in root:
        row = [elem.find(header).text if elem.find(header) is not None else '' for header in headers]
        writer.writerow(row)
