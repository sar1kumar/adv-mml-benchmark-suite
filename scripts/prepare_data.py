import argparse
import os
import requests
import zipfile
import shutil
import tqdm

def download_file(url, destination_folder, chunk_size=8192):
    """Downloads a file from a URL to a destination folder."""
    os.makedirs(destination_folder, exist_ok=True)
    local_filename = os.path.join(destination_folder, url.split('/')[-1] if url.split('/')[-1] else "downloaded_file")
    
    if "?zip=" in url:
        try:
            resource_id = url.split('/resources/')[1].split('/')[0]
            local_filename = os.path.join(destination_folder, f"{resource_id}.zip")
        except IndexError:
            local_filename = os.path.join(destination_folder, "downloaded_archive.zip")


    print(f"Downloading from {url} to {local_filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(local_filename, 'wb') as f, tqdm.tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    bar.update(size)
        print(f"Downloaded {local_filename} successfully.")
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_zip(zip_filepath, extract_to_folder):
    """Extracts a zip file to a specified folder."""
    print(f"Extracting {zip_filepath} to {extract_to_folder}...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_folder)
        print(f"Extracted {zip_filepath} successfully to {extract_to_folder}.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_filepath} is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"Error extracting {zip_filepath}: {e}")
        return False

def prepare_vqa_rad_data(output_base_dir):
    """Prepares data for the VQA-RAD task."""
    print(f"Preparing VQA-RAD data in {output_base_dir}...")
    vqa_rad_url = "https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip="
    
    # Create a temporary download directory
    temp_download_dir = os.path.join(output_base_dir, "temp_download_vqa_rad")
    os.makedirs(temp_download_dir, exist_ok=True)

    zip_filename = download_file(vqa_rad_url, temp_download_dir)
    
    if not zip_filename:
        print("Failed to download VQA-RAD data. Aborting.")
        shutil.rmtree(temp_download_dir)
        return

    # Create the final VQA-RAD data directory
    vqa_rad_data_dir = output_base_dir # User specifies the final data/vqa_rad dir
    os.makedirs(vqa_rad_data_dir, exist_ok=True)

    # Extract directly into the final directory
    if not extract_zip(zip_filename, vqa_rad_data_dir):
        print("Failed to extract VQA-RAD data. Aborting.")
        shutil.rmtree(temp_download_dir)
        return
        
    print("VQA-RAD data preparation complete.")
    print(f"Data should be in: {vqa_rad_data_dir}")
    print("Expected contents: 'VQA_RAD Dataset Public.json' and an 'images' folder.")

    # Clean up the downloaded zip file
    os.remove(zip_filename)
    # Remove the temporary download directory if it's empty
    if not os.listdir(temp_download_dir):
        os.rmdir(temp_download_dir)
    elif temp_download_dir != vqa_rad_data_dir : # only remove if it's a separate temp dir
        pass

def prepare_sme_data(output_base_dir):
    """Prepares data for the SME task."""
    print(f"Preparing SME data in {output_base_dir}...")
    sme_images_url = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    sme_questions_url = "https://github.com/LivXue/FS-MEVQA/raw/refs/heads/main/dataset/dataset.zip"
    # Create a temporary download directory
    temp_download_dir = os.path.join(output_base_dir, "temp_download_sme")
    os.makedirs(temp_download_dir, exist_ok=True)

    zip_filename = download_file(sme_images_url, temp_download_dir)
    
    if not zip_filename:
        print("Failed to download SME data. Aborting.")
        shutil.rmtree(temp_download_dir)
        return

    # Create the final SME data directory
    sme_data_dir = output_base_dir # User specifies the final data/sme dir
    os.makedirs(sme_data_dir, exist_ok=True)

    # Extract directly into the final directory
    if not extract_zip(zip_filename, sme_data_dir):
        print("Failed to extract SME data. Aborting.")
        shutil.rmtree(temp_download_dir)
        return
        
    print("SME data preparation complete.")
    print(f"Data should be in: {sme_data_dir}")
    print("Expected contents: 'images' folder with SME dataset images.")

    # Clean up the downloaded zip file
    os.remove(zip_filename)
    # Remove the temporary download directory if it's empty
    if not os.listdir(temp_download_dir):
        os.rmdir(temp_download_dir)
    elif temp_download_dir != sme_data_dir: # only remove if it's a separate temp dir
        pass


def main():
    parser = argparse.ArgumentParser(description="Download and prepare benchmark datasets.")
    parser.add_argument("--task", type=str, required=True, choices=["vqa_rad", "sme", "all"], # Add more tasks as needed
                        help="The specific task to prepare data for, or 'all'.")
    parser.add_argument("--output", type=str, required=True,
                        help="The base directory where the task-specific data folder will be created (e.g., data/). The script will create a subfolder named after the task (e.g. data/vqa_rad).")

    args = parser.parse_args()

    # The user provides a base path like 'data/', and the script creates 'data/vqa_rad/'
    # Or if user provides 'data/vqa_rad', that's also fine.
    # Let's ensure the output path refers to the specific task folder.
    
    task_output_dir = args.output # The user will specify the full path like 'data/vqa_rad'

    if not os.path.isdir(task_output_dir):
        print(f"Output directory {task_output_dir} will be created.")
    # os.makedirs(task_output_dir, exist_ok=True) # `prepare_vqa_rad_data` will handle its specific dir

    if args.task == "vqa_rad":
        prepare_vqa_rad_data(task_output_dir)
    elif args.task == "sme":
        prepare_sme_data(task_output_dir)
    elif args.task == "all":
        print("Preparing all datasets...")
        # Call each preparation function
        prepare_vqa_rad_data(os.path.join(args.output, "vqa_rad")) # Example: creates data/vqa_rad if output is data/
        prepare_sme_data(os.path.join(args.output, "sme"))
        # Add other tasks here: prepare_other_task_data(os.path.join(args.output, "other_task"))
        print("All dataset preparation finished.")
    else:
        print(f"Unknown task: {args.task}. Please choose from the available tasks.")

if __name__ == "__main__":
    main()