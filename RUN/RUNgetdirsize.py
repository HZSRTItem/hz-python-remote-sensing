import os
import sys


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size


def convert_bytes(size_in_bytes):
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024

    if size_in_gb >= 1:
        return f"{size_in_gb:.2f} GB"
    elif size_in_mb >= 1:
        return f"{size_in_mb:.2f} MB"
    elif size_in_kb >= 1:
        return f"{size_in_kb:.2f} KB"
    else:
        return f"{size_in_bytes} bytes"


def usage():
    print("srt_dirsize [opt: dirname] [opt: --sf] [--help]")


def RUNgetdirsize_main(argv):
    is_run = True
    folder_path = "."
    is_show_full = False
    i = 1
    while i < len(argv):
        if argv[i] == "--help":
            usage()
            is_run = False
        if argv[i] == "--sf":
            is_show_full = True
        else:
            folder_path = argv[i]
        i += 1
    folder_path = os.path.abspath(folder_path)
    if is_run:
        if not os.path.isdir(folder_path):
            is_run = False
            print("Can not find folder: " + folder_path)
    if is_run:
        total_size = 0
        len_max = 0

        for f in os.listdir(folder_path):
            ff = os.path.join(folder_path, f)
            if os.path.isdir(ff):
                if is_show_full:
                    if len_max < len(ff):
                        len_max = len(ff)
                else:
                    if len_max < len(f):
                        len_max = len(f)
        fmt = "{0:" + str(len_max) + "} {1}"
        for f in os.listdir(folder_path):
            ff = os.path.join(folder_path, f)
            if os.path.isdir(ff):
                folder_size = get_folder_size(ff)
                total_size += folder_size
                if is_show_full:
                    print(fmt.format(ff, convert_bytes(folder_size)))
                else:
                    print(fmt.format(f, convert_bytes(folder_size)))
            else:
                total_size += os.path.getsize(ff)

        if is_show_full:
            print(f"{folder_path} -> {convert_bytes(total_size)}")
        else:
            print(f"{os.path.split(folder_path)[1]} -> {convert_bytes(total_size)}")


if __name__ == "__main__":
    # RUNgetdirsize.py
    RUNgetdirsize_main(sys.argv)
