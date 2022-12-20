import zipfile
import os

def main():
    file = '/ssd/Cityscapes/leftImg8bit_sequence_trainvaltest.zip'
    archive = zipfile.ZipFile(file, 'r')
    namelist = sorted(archive.namelist())

    if os.path.exists(os.path.join('..', 'splits', 'cityscape')):
        print('path exists')
    else:
        os.makedirs(os.path.join('..', 'splits', 'cityscape'))
    with open(os.path.join('..', 'splits', 'cityscape', 'train.txt'), 'w') as trainfile:
        with open(os.path.join('..', 'splits', 'cityscape', 'val.txt'), 'w') as valfile:
            with open(os.path.join('..', 'splits', 'cityscape', 'test.txt'), 'w') as testfile:
                for i in range(len(namelist)):
                    str = namelist[i]
                    if 'png' in str:
                        if 'train' in str:
                            trainfile.write(str)
                            trainfile.write('\n')
                        elif 'val' in str:
                            valfile.write(str)
                            valfile.write('\n')
                        elif 'test' in str:
                            testfile.write(str)
                            testfile.write('\n')



if __name__ == '__main__':
    main()