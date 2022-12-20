# MIT License
#
# Copyright (c) 2020 Marvin Klingner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys

import dataloader.file_io.get_path as gp


class DirLister:
    """ This class will provide methods that enable the creation
        file lists and may return them in desired formats"""
    def __init__(self):
        pass

    @staticmethod
    def check_formats(cur_dir=None, file_ending=None):
        """ method to check if specified parameters have the right format

        :param cur_dir: directory which is checked for existance
        :param file_ending: file ending that is checked for the right format
        """
        check = True
        if cur_dir is not None:
            if os.path.isdir(cur_dir) is False:
                print("the specified directory does not exist")
                check = False
        if file_ending is not None:
            if file_ending[0] != '.':
                print("the file ending has no '.' at the beginning")
                check = False
        return check

    @staticmethod
    def list_subdirectories(top_dir):
        """ method that lists all subdirectories of a given directory

        :param top_dir: directory in which the subdirectories are searched in
        """
        top_dir = os.path.abspath(top_dir)
        sub_dirs = [os.path.join(top_dir, x) for x in os.listdir(top_dir)
                    if os.path.isdir(os.path.join(top_dir, x))]
        return sub_dirs

    @staticmethod
    def list_files_in_directory(top_dir):
        """ method that lists all files of a given directory

        :param top_dir: directory in which the files are searched in
        """
        top_dir = os.path.abspath(top_dir)
        files = [os.path.join(top_dir, x) for x in os.listdir(top_dir)
                 if os.path.isfile(os.path.join(top_dir, x))]
        return files

    @staticmethod
    def get_directories(parent_dir):
        """ method that lists all directories of a given directory recursively

        :param parent_dir: directory in which the subdirectories are searched in
        """
        if DirLister.check_formats(cur_dir=parent_dir) is False:
            sys.exit("Inputparameter überprüfen")
        parent_dir = os.path.abspath(parent_dir)
        sub_dirs = []
        still_to_search = DirLister.list_subdirectories(parent_dir)

        while len(still_to_search) > 0:
            curr_sub_dirs = DirLister.list_subdirectories(still_to_search[0])
            if len(curr_sub_dirs) == 0:
                sub_dirs.append(still_to_search[0])
            else:
                still_to_search.extend(curr_sub_dirs)
            still_to_search.remove(still_to_search[0])

        return sub_dirs

    @staticmethod
    def include_files_by_name(file_list, names, positions):
        """ takes a list of filepaths and keeps only the files which have all strings
        inside of their path specified by the list names

        :param file_list: list of filepaths
        :param names: strings which have to be inside the directory name
        :param positions: positions inside the dataset which are also only kept if the element is kept
        """
        if type(names) == list:
            for name in names:
                positions = [positions[i] for i in range(len(file_list)) if name in file_list[i]]
                file_list = [x for x in file_list if name in x]
        elif type(names) == str:
            name = names
            positions = [positions[i] for i in range(len(file_list)) if name in file_list[i]]
            file_list = [x for x in file_list if name in x]
        return file_list, positions

    @staticmethod
    def include_files_by_folder(file_list, names, positions):
        """ takes a list of filepaths and keeps only the files which have all strings
        inside of their path specified by the list names

        :param file_list: list of filepaths
        :param names: folders which have to be inside the directory path
        :param positions: positions inside the dataset which are also only kept if the element is kept
        """
        if type(names) == list:
            for name in names:
                positions = [positions[i] for i in range(len(file_list)) if name in file_list[i]]
                file_list = [x for x in file_list if name + os.sep in x or name == os.path.split(x)[1]]
        elif type(names) == str:
            name = names
            positions = [positions[i] for i in range(len(file_list)) if name in file_list[i]]
            file_list = [x for x in file_list if name + os.sep in x or name == os.path.split(x)[1]]
        return file_list, positions

    @staticmethod
    def include_dirs_by_name(dir_list, names, ignore=(), ambiguous_names_to_ignore=()):
        """ takes a list of directories and includes the directories which have all strings
        of the ones specified by the list names

        :param dir_list: list of directories
        :param names: strings which have to be inside the directory name
        :param ignore: string that must not be inside the directory name
        :param ambiguous_names_to_ignore: A list containing all strings that should not be taken into account when
            comparing to names. For example, if an upper folder is called 'dataset_images' and one filter name
            is also 'images' (e.g. for the color image), then this parameter will prevent all folder from being
            returned
        :return: a list of all folders containing all names, excluding those containing a string in ignore
        """
        shortened_dir_list = dir_list.copy()
        if type(ambiguous_names_to_ignore) == str:
            ambiguous_names_to_ignore = (ambiguous_names_to_ignore, )
        for ambiguous_name in ambiguous_names_to_ignore:
            shortened_dir_list = [x.replace(ambiguous_name, '') for x in shortened_dir_list]
        if type(names) == list:
            for name in names:
                dir_list = [x for x, xs in zip(dir_list, shortened_dir_list) if name in xs]
                shortened_dir_list = [xs for xs in shortened_dir_list if name in xs]
        elif type(names) == str:
            name = names
            dir_list = [x for x, xs in zip(dir_list, shortened_dir_list) if name in xs]
        for ignore_string in ignore:
            dir_list = [x for x in dir_list if ignore_string not in x]
        return dir_list

    @staticmethod
    def include_dirs_by_folder(dir_list, names):
        """ takes a list of directories and includes the directories which have all strings
        of the ones specified by the list names

        :param dir_list: list of directories
        :param names: folders which have to be inside the directory path
        """
        if type(names) == list:
            for name in names:
                dir_list = [x for x in dir_list if name + os.sep in x or name == os.path.split(x)[1]]
        elif type(names) == str:
            name = names
            dir_list = [x for x in dir_list if name + os.sep in x or name == os.path.split(x)[1]]
        return dir_list

    @staticmethod
    def remove_dirs_by_name(dir_list, names):
        """ takes a list of directories and removes the directories which have at least one string
        of the ones specified by the list names

        :param dir_list: list of directories
        :param names: strings which are not allowed inside the directory name
        """
        if type(names) == list:
            for name in names:
                dir_list = [x for x in dir_list if name not in x]
        elif type(names) == str:
            name = names
            dir_list = [x for x in dir_list if name not in x]
        return dir_list

    @staticmethod
    def get_files_by_ending(cur_dir, file_ending, ignore = []):
        """ returns all files inside a directory which have a certain ending

        :param cur_dir: list of directories
        :param file_ending: all files with the specified file_ending are returned
        :param ignore: list of strings. Filenames containing one of these strings will be ignored.
        :return: all files inside cur_dir which have the ending file_ending
        """
        if DirLister.check_formats(cur_dir=cur_dir,
                                   file_ending=file_ending) is False:
            sys.exit("Inputparameter überprüfen")
        files = DirLister.list_files_in_directory(cur_dir)
        len_ending = len(file_ending)
        files = [x for x in files if x[-len_ending:] == file_ending]
        for ignore_string in ignore:
            files = [x for x in files if ignore_string not in x]
        return files


if __name__ == '__main__':
    """can be used for testing purposes"""
    path_getter = gp.GetPath()
    path = path_getter.get_data_path()
    path = os.path.join(path,  'Cityscapes')
    a = DirLister()
    test = a.get_directories(path)
    print(a.include_dirs_by_name(test, 'test'))
