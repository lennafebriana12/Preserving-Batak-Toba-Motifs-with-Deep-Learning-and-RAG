import os

def print_folder_structure(path=".", indent=0):
    """Menampilkan struktur folder di dalam proyek tanpa file."""
    # Menampilkan nama folder
    if os.path.isdir(path):
        print("  " * indent + os.path.basename(path))
        # Melanjutkan ke subfolder
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):  # Hanya folder yang ditampilkan
                print_folder_structure(full_path, indent + 1)

# Menjalankan fungsi untuk menampilkan folder di dalam proyek
print_folder_structure(os.getcwd())