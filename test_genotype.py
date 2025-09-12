import os
import argparse
import time
import re
from scipy.io import savemat
from torch.utils.data import DataLoader
import distributed_utils
from my_dataset import *
from utils import model_selection, plot_confusion_matrix


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main_test(args):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_path = args.data_path
    save_path = args.save_path

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if os.path.exists(save_path + r"/result") is False:
        os.makedirs(save_path + r"/result")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print("using {} device.".format(device))
    print('')

    # classification num
    num_classes = args.num_classes

    test_dataset, test_images_path = dataset_test(path=data_path,
                                                  r_seed=args.random_seed,
                                                  channel=args.channels)
    print("Total test dataset is {}.".format(len(test_dataset)))

    # print(test_dataset.images_path)
    batch_size = args.batch_size
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    print('')

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=nw,
                             collate_fn=test_dataset.collate_fn)

    # 载入训练好的模型
    model = model_selection(model_name=args.model_name,
                            num_class=num_classes,
                            in_channel=len(args.channels),
                            drop_rate=args.drop_rate,
                            device=device,
                            patch_size=args.patch_size)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)['model']
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
        print('')

    # test
    sample_num = 0
    pred_path = []
    model.eval()

    confmat = distributed_utils.ConfusionMatrix(num_classes)
    dice = distributed_utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad(), open(save_path + '/predictions.txt', 'w') as file:
        for images, targets in metric_logger.log_every(test_loader, 1, header):
            sample_num += images.shape[0]  # images.shape[0]即为每次载入的batch_size
            images, targets = images.to(device), targets.to(device)

            # init model
            img_height, img_width = images.shape[-2:]
            init_img = torch.zeros((1, len(args.channels), img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(images)
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            pred_softmax = torch.softmax(output['out'], dim=1)  # 预测概率值
            pred_classes = output['out'].argmax(1)  # 预测分类值
            confmat.update(targets.flatten(), output['out'].argmax(1).flatten())
            # dice
            dice.update(output['out'], targets)
            # 将预测类别和真实类别转换为CPU上的NumPy数组
            pred_softmax_np = pred_softmax.to("cpu").numpy()
            pred_classes_np = pred_classes.to("cpu").numpy()
            targets_np = targets.to("cpu").numpy()
            pred_classes = pred_classes_np.astype(np.uint8)

            # 正则表达式模式，用于匹配文件名中的数字
            pattern = re.compile(r'patch_(\d+)_(\d+)\.mat')
            for i in range(len(images)):
                path = test_images_path[sample_num - len(images) + i]
                pred_path.append(test_images_path[sample_num - len(images) + i])
                match = pattern.search(path)
                if match:
                    row, col = match.groups()
                    mat_file_path = os.path.join(save_path, f'result/patch_{row}_{col}_pred.mat')
                    savemat(mat_file_path, {'pred_classes': pred_classes_np[i],
                                            'pred_softmax': pred_softmax_np[i],
                                            'labels': targets_np[i]})

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        conf_matrix = np.array(confmat.mat.cpu())

        IDHpred_num = conf_matrix.sum()
        IDHpred_correctnum = conf_matrix.diagonal(offset=0).sum()
        IDH_Acc = IDHpred_correctnum / IDHpred_num
        IDH_Sen = conf_matrix[0, 0] / conf_matrix[0, :].sum()
        IDH_Spe = conf_matrix[1, 1] / conf_matrix[1, :].sum()
        IDH_FPR = 1 - IDH_Spe
        IDH_TPR = IDH_Sen
        IDH_Recall = IDH_Sen
        IDH_Pre = conf_matrix[0, 0] / conf_matrix[:, 0].sum()
        IDH_F1 = 2 * IDH_Recall * IDH_Pre / (IDH_Recall + IDH_Pre)

        test_info = str(confmat)
        print(test_info)
        file.write(test_info + "\n\n")

        file.write(
            "ConfusionMatrix_IDH:\n                       Pred label\n                      IDH0    IDH1\n")
        file.write(
            "        True    IDH0  {:.0f},   {:.0f}\n".format(conf_matrix[0, 0], conf_matrix[0, 1]))
        file.write(
            "        label   IDH1  {:.0f},   {:.0f}\n\n".format(conf_matrix[1, 0], conf_matrix[1, 1]))

        file.write("IDH Accuracy: {:.5f}%\n".format(IDH_Acc * 100))
        file.write("IDH Sensitivity: {:.5f}%\n".format(IDH_Sen * 100))
        file.write("IDH Specificity: {:.5f}%\n".format(IDH_Spe * 100))
        file.write("IDH Recall: {:.5f}%\n".format(IDH_Recall * 100))
        file.write("IDH Precision: {:.5f}%\n".format(IDH_Pre * 100))
        file.write("IDH F1 Score: {:.5f}%\n".format(IDH_F1 * 100))
        file.write("Whole Dice: {:.5f}%\n".format(dice.value.item() * 100))

    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix, labels_names=['IDH0', 'IDH1'],
                          save_path=save_path + '/result/confusion_matrix.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=289)  # IDH 144  MGMT 289
    parser.add_argument('--channels', type=list, default=list(range(41)))
    parser.add_argument('--patch-size', type=int, default=48)  # IDH 64  MGMT 48

    parser.add_argument('--IDH-type', type=str, default=r"MGMT0_11")  # 动态赋值
    parser.add_argument('--data-path', type=str, default=r"./data/MGMT_patch")
    parser.add_argument('--save-path', type=str, default=r"./result/MGMT_genotype")
    parser.add_argument('--model_name', type=str, default="mixunet")
    parser.add_argument('--weights', type=str, default=r'weights/MGMT_genotype.pth',
                        help='initial weights path')
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--random_seed', default=1)
    parser.add_argument('--torch_seed', default=1)

    opt = parser.parse_args()

    # 调用你的测试函数
    main_test(opt)
