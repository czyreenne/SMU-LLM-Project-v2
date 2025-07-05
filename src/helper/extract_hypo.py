# ----- IMPORTS -----

import re
import fitz  # PyMuPDF
import json
import os
import argparse
import logging
from datetime import datetime
import time

# ----- HELPER FUNCTIONS -----

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_content(pdf_path):
    start_time = time.time()
    logging.info(f"Starting extraction for {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF {pdf_path}: {str(e)}")
        return None, None, None
    num_pages = doc.page_count
    scenario = ""
    questions = []
    if num_pages > 1:
        last_page_text = doc.load_page(num_pages - 1).get_text()
        questions = re.findall(r'\d+\..*?(?=\d+\.|$)', last_page_text, re.S)
        if questions:
            last_qn = questions.pop()
            end_of_qn = last_qn.find('?')
            trimmed_qn = last_qn[:end_of_qn + 1] if end_of_qn != -1 else last_qn
            questions.append(trimmed_qn)
        scenario_text_pages = []
        for i in range(1, num_pages):
            page_text = doc.load_page(i).get_text()
            cut_index = page_text.find("* * *")
            if cut_index != -1:
                page_text = page_text[cut_index + 5:]
            scenario_text_pages.append(page_text)
        if questions:
            first_question_start = last_page_text.find(questions[0])
            scenario_text_pages.append(last_page_text[:first_question_start])
        scenario = " ".join(scenario_text_pages).strip()
    doc.close()
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Extraction completed for {pdf_path}. Duration: {duration:.2f} seconds")
    return scenario, [question.strip() for question in questions], {
        'extraction_time': datetime.now().isoformat(),
        'duration': duration,
        'num_pages': num_pages,
        'num_characters': len(scenario),
        'num_words': len(scenario.split()),
    }

def main(inpath, outpath):
    setup_logging()
    pdf_files = [os.path.join(inpath, f) for f in os.listdir(inpath) if f.endswith('.pdf')]
    results = []
    for pdf_file in pdf_files:
        scenario, questions, metadata = extract_content(pdf_file)
        if scenario is not None and questions is not None:
            data = {
                'metadata': metadata,
                'file': os.path.basename(pdf_file),
                'scenario': scenario,
                'questions': questions,
            }
            results.append(data)
            file_name = os.path.splitext(os.path.basename(pdf_file))[0] # im writing to individual json files here, but can remove if that's being extra ~ gong
            individual_output_path = os.path.join(outpath, f"{file_name}_extracted.json")
            with open(individual_output_path, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Individual result saved to {individual_output_path}")
    os.makedirs(outpath, exist_ok=True)
    output_file_path = os.path.join(outpath, 'extracted_data.json') # original combined JSON file 
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Combined results saved to {output_file_path}")
    print(f"Extraction completed. Results saved to {outpath}")

# ----- SAMPLE EXECUTION CODE -----

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract paragraphs from PDF files.")
    parser.add_argument('--inpath', type=str, default='data/raw', help='Input directory containing PDF files.')
    parser.add_argument('--outpath', type=str, default='data/processed', help='Output directory for JSON results.')
    args = parser.parse_args()
    main(args.inpath, args.outpath)