import os
from os.path import join as pjoin


def get_nonexistent_path(fname_path):
    """Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    get_nonexistant_path('/etc/issue')
    '/etc/issue_1'

    Args:
        fname_path: string, path to increment

    Returns:
        non existent file path
    """
    if not os.path.exists(fname_path):
        os.makedirs(fname_path)
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    os.mkdir(new_fname)
    return new_fname


def create_model_directories(name, subdirectories=[]):
    """ Create the folder architecture to save everything.

    Args:
        name: string, name of the parent directory
        subdirectories: tuple of strings, name and names of the subdirectories

    Returns:
        tuple containing new name and subdirectory paths in same order as arg
    """

    name = get_nonexistent_path(name)

    subdir_paths = [name]

    for subdir in subdirectories:
        if not os.path.exists(pjoin(name, subdir)):
            os.mkdir(pjoin(name, subdir))

        subdir_paths.append(pjoin(name, subdir))

    return tuple(subdir_paths)
