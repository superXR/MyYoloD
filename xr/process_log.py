import os
from tensorboardX import SummaryWriter

log_path_list = ['/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/v4.2_2022-03-27-15-33/_2022-03-27-15-33_train.log'
                ]

out_path = '/mnt/sdb/dpai3/project/YOLOP/tools/runs/BddDataset/v4.2_2022-03-27-15-33/metrics'


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
                result[tmp_epoch]['ll_Acc'] = line_list[3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['ll_IOU'] = line_list[-3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['ll_mIOU'] = line_list[-1].split('(')[-1].split(')')[0]
                continue
            if 'Driving' in line:
                line_list = line.split(' ')
                result[tmp_epoch]['da_Acc'] = line_list[3].split('(')[-1].split(')')[0]
                result[tmp_epoch]['da_IOU'] = line_list[-5].split('(')[-1].split(')')[0]
                result[tmp_epoch]['da_mIOU'] = line_list[-1].split('(')[-1].split(')')[0]

        f.close()
    return result


# def make_events(out_path):
#     writer = SummaryWriter(out_path)

#     writer.close()


def main():
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    writer = SummaryWriter(out_path)
    result = extract(log_path_list)
    # [epoch, best value]
    best_precision = [0.0, 0]
    best_recall = [0.0, 0]
    best_map_50 = [0.0, 0]
    ll_best_acc = [0.0, 0]
    ll_best_IoU = [0.0, 0]
    ll_best_mIoU = [0.0, 0]
    da_best_acc = [0.0, 0]
    da_best_IoU = [0.0, 0]
    da_best_mIoU = [0.0, 0]
    for k,v in result.items():
        if v == {}:
            continue
        print(k, v)
        # writer.add_scalar('P_R', float(v['R']), 10.0 * float(v['P']))
        # detect metrics
        if float(v['P']) > best_precision[0]:
            best_precision = [float(v['P']), int(k)]
        if float(v['mAP@0.5']) > best_map_50[0]:
            best_map_50 = [float(v['mAP@0.5']), int(k)]
        if float(v['R']) > best_recall[0]:
            best_recall = [float(v['R']), int(k)]
        # lane line metrics
        if float(v['ll_Acc']) > ll_best_acc[0]:
            ll_best_acc = [float(v['ll_Acc']), int(k)]
        if float(v['ll_IOU']) > ll_best_IoU[0]:
            ll_best_IoU = [float(v['ll_IOU']), int(k)]
        if float(v['ll_mIOU']) > ll_best_mIoU[0]:
            ll_best_mIoU = [float(v['ll_mIOU']), int(k)]
        # drivable area metrics
        if float(v['da_Acc']) > da_best_acc[0]:
            da_best_acc = [float(v['da_Acc']), int(k)]
        if float(v['da_IOU']) > da_best_IoU[0]:
            da_best_IoU = [float(v['da_IOU']), int(k)]
        if float(v['da_mIOU']) > da_best_mIoU[0]:
            da_best_mIoU = [float(v['da_mIOU']), int(k)]
        writer.add_scalars('Detect/P_R', {'Precision':10.0 * float(v['P']), 'Recall':float(v['R'])}, int(k))
        writer.add_scalar('Detect/mAP@0.5', float(v['mAP@0.5']), int(k))
        writer.add_scalar('Detect/mAP@0.5:0.95', float(v['mAP@0.5:0.95']), int(k))
        writer.add_scalar('ll/Acc', float(v['ll_Acc']), int(k))
        writer.add_scalar('ll/IOU', float(v['ll_IOU']), int(k))
        writer.add_scalar('ll/mIOU', float(v['ll_mIOU']), int(k))
        writer.add_scalar('da/Acc', float(v['da_Acc']), int(k))
        writer.add_scalar('da/IOU', float(v['da_IOU']), int(k))
        writer.add_scalar('da/mIOU', float(v['da_mIOU']), int(k))
    writer.close()
    print('best_precision:', best_precision)
    print('best_recall:', best_recall)
    print('best_map_50:', best_map_50)
    print('ll_best_acc:', ll_best_acc)
    print('ll_best_IoU:', ll_best_IoU)
    print('ll_best_mIoU:', ll_best_mIoU)
    print('da_best_acc:', da_best_acc)
    print('da_best_IoU:', da_best_IoU)
    print('da_best_mIoU:', da_best_mIoU)
    return 0


if __name__ == '__main__':
    main()