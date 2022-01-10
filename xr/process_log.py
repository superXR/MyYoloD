import os
from tensorboardX import SummaryWriter

log_path_list = ['/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/_2021-12-28-23-14/_2021-12-28-23-14_train.log',
                '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/_2022-01-03-19-28/_2022-01-03-19-28_train.log'
                ]

out_path = '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/baseline'


def extract(log_path_list):
    result = dict()
    tmp_epoch = '1'
    for log_path in log_path_list:
        f = open(log_path, 'r')
        result[tmp_epoch] = dict()
        for line in f.readlines():
            if 'epoch-' in line:
                epoch = line.split('/')[-1].split('.')[0].split('-')[-1]
                tmp_epoch = str(int(epoch) + 1)
                result[tmp_epoch] = dict()
                continue
            if 'Detect' in line:
                line_list = line.split(' ')
                result[tmp_epoch]['P'] = line_list[1].split('(')[-1].split(')')[0]
                result[tmp_epoch]['R'] = line_list[3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['mAP@0.5'] = line_list[-3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['mAP@0.5:0.95'] = line_list[-1].split('(')[-1].split(')')[0]
                continue
            if 'Lane' in line:
                line_list = line.split(' ')
                result[tmp_epoch]['Acc'] = line_list[3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['IOU'] = line_list[-3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['mIOU'] = line_list[-1].split('(')[-1].split(')')[0]
                continue
        f.close()
    return result


# def make_events(out_path):
#     writer = SummaryWriter(out_path)

#     writer.close()


def main():
    writer = SummaryWriter(out_path)
    result = extract(log_path_list)
    for k,v in result.items():
        if v == {}:
            continue
        print(k, v)
        # writer.add_scalar('P_R', float(v['R']), 10.0 * float(v['P']))
        writer.add_scalars('Detect/P_R', {'Precision':10.0 * float(v['P']), 'Recall':float(v['R'])}, int(k))
        writer.add_scalar('Detect/mAP@0.5', float(v['mAP@0.5']), int(k))
        writer.add_scalar('Detect/mAP@0.5:0.95', float(v['mAP@0.5:0.95']), int(k))
        writer.add_scalar('Lane/Acc', float(v['Acc']), int(k))
        writer.add_scalar('Lane/IOU', float(v['IOU']), int(k))
        writer.add_scalar('Lane/mIOU', float(v['mIOU']), int(k))
    writer.close()
    return 0


if __name__ == '__main__':
    main()