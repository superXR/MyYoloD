## 处理pred结果的.json文件,画图
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False,palette=None,is_demo=False,is_gt=False,respective=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
                0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3 # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    
    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        # color_area = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        ll_color_area = color_area.copy()
        color_area[result[0] == 1] = [0, 255, 0]  # Driving area color
        da_seg = color_area.copy()
        try:
            color_area[result[1] ==1] = [255, 0, 0]  # Lane line color 
            ll_color_area[result[1] ==1] = [255, 0, 0]
            ll_seg = ll_color_area
        except:
            print(result[1].shape)
        color_seg = color_area

    if respective:
        da_img, ll_img = img.copy(), img.copy()
        # convert to BGR
        da_color_seg = da_seg[..., ::-1]
        da_color_mask = np.mean(da_color_seg, 2)
        # if img.shape[0] == 720:
        da_img[da_color_mask != 0] = da_img[da_color_mask != 0] * 0.5 + da_color_seg[da_color_mask != 0] * 0.5
        # img = img * 0.5 + color_seg * 0.5
        da_img = da_img.astype(np.uint8)
        da_img = cv2.resize(da_img, (1280,720), interpolation=cv2.INTER_LINEAR)

        # convert to BGR
        ll_color_seg = ll_seg[..., ::-1]
        ll_color_mask = np.mean(ll_color_seg, 2)
        # if img.shape[0] == 720:
        ll_img[ll_color_mask != 0] = ll_img[ll_color_mask != 0] * 0.5 + ll_color_seg[ll_color_mask != 0] * 0.5
        # img = img * 0.5 + color_seg * 0.5
        ll_img = ll_img.astype(np.uint8)
        ll_img = cv2.resize(ll_img, (1280,720), interpolation=cv2.INTER_LINEAR)

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    # if img.shape[0] == 720:
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    print(img.shape, '----', color_seg.shape)
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_LINEAR)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_segresult.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_segresult.png".format(epoch,index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir+"/batch_{}_{}_da_seg_gt.png".format(epoch,index), img)
            else:
                cv2.imwrite(save_dir+"/batch_{}_{}_ll_seg_gt.png".format(epoch,index), img)  
    if respective:
        return img,da_img,ll_img
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # print(label)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass
# def plot():
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU) if not cfg.DEBUG \
#         else select_device(logger, 'cpu')

#     if args.local_rank != -1:
#         assert torch.cuda.device_count() > args.local_rank
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device('cuda', args.local_rank)
#         dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    
#     model = get_net(cfg).to(device)
#     model_file = '/home/zwt/DaChuang/weights/epoch--2.pth'
#     checkpoint = torch.load(model_file)
#     model.load_state_dict(checkpoint['state_dict'])
#     if rank == -1 and torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
#     if rank != -1:
#         model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)