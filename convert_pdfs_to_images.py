import argparse
import glob
import os
import pdf2image
from tqdm import tqdm


def main(args):

    mask = os.path.join(args.src_root_folder, '**/*.pdf')
    file_list = glob.glob(mask, recursive=True)
    for pdf_file_path in tqdm(file_list):
        relative_pdf_file_path = os.path.relpath(pdf_file_path, args.src_root_folder)
        relative_pdf_file_folder, pdf_file_name = os.path.split(relative_pdf_file_path)
        output_images_folder = os.path.join(
            args.dst_root_folder, relative_pdf_file_folder)
        os.makedirs(output_images_folder, exist_ok=True)

        pdf2image.convert_from_path(
            pdf_file_path, dpi=72, fmt='jpg', 
            output_folder=output_images_folder,
            single_file=True, output_file=os.path.splitext(pdf_file_name)[0],
            paths_only=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_root_folder', help='Root folder, containing pdf files.')
    parser.add_argument('dst_root_folder', help='Destination folder, where images will be stored.')
    main(parser.parse_args())