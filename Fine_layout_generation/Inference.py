import os
import sys
import shutil
import subprocess


def clear_folder(folder_path):
    """
    Delete all files and subfolders in the specified folder (keep the folder itself).
    :param folder_path: Folder path (e.g., "./u")
    """
    # Check whether the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist; skipping cleanup.")
        return

    # Iterate over all folder contents
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            # If it is a file or symlink, delete it
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"Deleted file: {item_path}")
            # If it is a subfolder, delete it recursively (including its contents)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")


def run_py_scripts(script_paths):
    python_exe = sys.executable

    for script in script_paths:
        if not os.path.exists(script):
            print(f"⚠️ Script {script} does not exist; skipping.")
            continue

        print(f"\n========== Starting {script} ==========")

        try:
            # check=True: Raise an exception immediately if the script fails
            # text=True: is shorthand for encoding="utf-8"
            result = subprocess.run(
                [python_exe, script],
                check=True,
                #capture_output=True,  # Capture stdout and stderr automatically
                #text=True
            )
            #print(f"[Output]: \n{result.stdout}")
            print(f"✅ {script} completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"\n\033[1;31m❌ Script execution failed! Stopping the remaining tasks.\033[0m")
            print(f"[Error output]: \n{e.stderr}")
            sys.exit(1)  # Exit the main program


if __name__ == "__main__":
    # 1. Configure paths (adjust for your setup)
    # Scripts to run, in order
    scripts = [
        "predict_sample.py",
        "vectorization.py",
        "figure_redraw.py",
    ]
    # Folder to clear
    target_folder = ['../test/Fine_output']

    # 2. Clear the u folder (optional: this step can be moved to the end)
    print("========== Clearing the folder ==========")
    for folder in target_folder:
        clear_folder(folder)

    # 3. Run a.py → b.py → c.py in order
    run_py_scripts(scripts)
    #shutil.rmtree(f'../test/Fine_output/cache_img/')
    print("\n========== All operations completed ==========")
