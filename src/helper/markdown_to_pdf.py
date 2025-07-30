'''
Script to convert Markdown files to PDF format using Pandoc.
This script reads a Markdown file, processes it, and generates a PDF output.
'''
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
import argparse

def convert_markdown_to_pdf(input_file: Path, output_file: Path = None):
    """
    Converts a Markdown file to PDF using the markdown-pdf Python library.
    
    Args:
        input_file (Path): Path to the input Markdown file.
        output_file (Path, optional): Path to the output PDF file. If not specified, saves PDF in the same folder as the markdown file.
    """
    # Ensure the input file exists
    input_file = Path(input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"The input file {input_file} does not exist.")

    # If output_file is not specified, save PDF in the same folder as markdown
    if output_file is None:
        output_file = input_file.with_suffix('.pdf')
    else:
        output_file = Path(output_file)

    try:
        with open(input_file, 'r', encoding='utf-8') as inf:
            markdown_content = inf.read()
            pdf = MarkdownPdf()
            pdf.add_section(Section(markdown_content))
            pdf.save(str(output_file))
            print(f"Successfully converted {input_file} to {output_file}.")
    except Exception as e:
        print(f"An error occurred while converting: {e}")

def convert_dir(dir, nested):
    '''
    Converts all Markdown files in a directory (optionally recursively) to PDF format.
    The converted PDF files are saved in the same folder as the input Markdown files.
    '''
    dir = Path(dir)
    if not dir.is_dir():
        raise NotADirectoryError(f"The path {dir} is not a directory.")
    if nested:
        md_files = dir.rglob("*.md")
    else:
        md_files = dir.glob("*.md")
    for file in md_files:
        output_file = file.with_suffix('.pdf')
        convert_markdown_to_pdf(file, output_file)
        print(f"Converted {file} to {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result Markdown to PDF Converter")
    parser.add_argument("--inf", type=str, required=True, help="Path to directory with results.md files or single file")
    parser.add_argument("--outf", type=str, default=None, help="Path to output pdf file")
    parser.add_argument("--nested", action='store_true', help="If directory supplied, convert Markdown files in subdirectories as well")
    args = parser.parse_args()
    inpath = Path(args.inf)
    if inpath.is_dir():
        convert_dir(args.inf, args.nested)
    else:
        convert_markdown_to_pdf(args.inf, args.outf)

