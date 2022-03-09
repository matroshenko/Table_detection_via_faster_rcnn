import argparse
import glob
import os
import pdf2image
from PyPDF2 import PdfFileReader
from tqdm import tqdm


def get_pdf_file_shape(pdf_file_name):
    with open(pdf_file_name, 'rb') as pdf_file:
        pdf_page = PdfFileReader(pdf_file).getPage(0)
        pdf_shape = pdf_page.mediaBox
        pdf_height = float(pdf_shape[3]-pdf_shape[1])
        pdf_width = float(pdf_shape[2]-pdf_shape[0])
    return pdf_height, pdf_width

def main(args):

    mask = os.path.join(args.src_root_folder, '**/*.pdf')
    file_list = glob.glob(mask, recursive=True)
    for pdf_file_path in tqdm(file_list):
        relative_pdf_file_path = os.path.relpath(pdf_file_path, args.src_root_folder)
        relative_pdf_file_folder, pdf_file_name = os.path.split(relative_pdf_file_path)
        output_images_folder = os.path.join(
            args.dst_root_folder, relative_pdf_file_folder)
        os.makedirs(output_images_folder, exist_ok=True)

        pdf_height, pdf_width = get_pdf_file_shape(pdf_file_path)
        pdf2image.convert_from_path(
            pdf_file_path, size=(pdf_width, pdf_height), fmt='jpg', 
            output_folder=output_images_folder,
            single_file=True, output_file=os.path.splitext(pdf_file_name)[0],
            paths_only=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_root_folder', help='Root folder, containing pdf files.')
    parser.add_argument('dst_root_folder', help='Destination folder, where images will be stored.')
    main(parser.parse_args())