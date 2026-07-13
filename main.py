from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

file_path = "/Users/shivanshmahajan/Developer/Document_Intelligence/test_files/NIPS-2017-attention-is-all-you-need-Paper.pdf"
base_file_name="gandu"
def main():
    elements = partition_pdf(filename=file_path)
    elements_to_json(elements=elements, filename=f"{file_path}/{base_file_name}-output.json")

if __name__ == "__main__":
    main()