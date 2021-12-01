import os


# Returns the absolute directory that this file is in.
# Useful for adding to relative paths when you put thi file in the project directory.
def this_directory():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return this_dir


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def get_file_names_in_directory(dir_name):
    return os.listdir(dir_name)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
