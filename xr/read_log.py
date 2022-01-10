import os


log_path = '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/_2022-01-03-19-28/_2022-01-03-19-28_train.log'
log_dir = '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/_2022-01-03-19-28/'
best_map_file = '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/_2022-01-03-19-28/map_tmp.txt'

def main():
    num = 0
    epoch = 0
    map_50 =0
    best_map = 0
    f = open(log_path, 'r')
    for line in f.readlines()[::-1]:
        if 'epoch-' in line:
            epoch = line.split('/')[-1].split('.')[0].split('-')[-1]
            if int(epoch) % 10 == 0:
                cmd = 'cp ' + log_dir + 'checkpoint.pth ' + log_dir + 'epoch-' + epoch + '.pth'
                os.system(cmd)
        if 'mAP' in line:
            map_50 = line.split(' ')[-3].split('(')[-1].split(')')[0]
            with open(best_map_file, 'r') as ftmp:
                best_map = ftmp.readline().split('\n')[0]
            if map_50 > best_map:
                cmd = 'cp ' + log_dir + 'checkpoint.pth ' + log_dir + 'best.pth'
                os.system(cmd)
                with open(best_map_file, 'w') as fp:
                    fp.write(map_50)
                print('update best model success epoch:', epoch)
        num += 1
        if num >= 79:
            break
    f.close()

if __name__ == '__main__':
    main()