if __name__ == "__main__":

    f = open('val_files.txt', 'w')
    for i in range(108):
        f.writelines(['2011_09_26/2011_09_26_drive_0001_sync ', str(i).zfill(10), ' l\n'])

    f.close()
    print('done')