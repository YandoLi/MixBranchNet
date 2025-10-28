import os
import argparse
from scipy.io import savemat
from torch.utils.data import DataLoader
import distributed_utils
from my_dataset import *
from utils import model_selection


def main_test(args):
    data_path = args.data_path
    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))
    print('')

    num_classes = args.num_classes + 1  # segmentation: nun_classes + background

    test_dataset, test_images_path = dataset_test(path=data_path,
                                                  r_seed=args.random_seed,
                                                  channel=args.channels)
    print("Total test dataset is {}.".format(len(test_dataset)))

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

    # select model
    model = model_selection(model_name=args.model_name,
                            num_class=num_classes,
                            in_channel=len(args.channels),
                            drop_rate=args.drop_rate,
                            device=device,
                            patch_size=args.image_size)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)['model']
            print(model.load_state_dict(weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
        print('')

    # test
    sample_num = 0
    model.eval()

    confmat = distributed_utils.ConfusionMatrix(num_classes)
    dice = distributed_utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = distributed_utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(test_loader, 1, header):
            sample_num += images.shape[0]  # images.shape[0]: batch_size
            images, targets = images.to(device), targets.to(device)

            # init model
            img_height, img_width = images.shape[-2:]
            init_img = torch.zeros((1, len(args.channels), img_height, img_width), device=device)
            model(init_img)
            # prediction
            output = model(images)
            pred_softmax = torch.softmax(output['out'], dim=1)  # predicted probabilities
            pred_classes = output['out'].argmax(1)  # predicted classes
            confmat.update(targets.flatten(), output['out'].argmax(1).flatten())
            # convert to numpy
            pred_softmax_np = pred_softmax.to("cpu").numpy()
            pred_classes_np = pred_classes.to("cpu").numpy()
            targets_np = targets.to("cpu").numpy()

            # dice
            dice.update(output['out'], targets)

            for i in range(len(images)):
                mat_file_path = os.path.join(save_path, f'pred_mask.mat')
                savemat(mat_file_path, {'pred_classes': pred_classes_np[i],
                                        'pred_softmax': pred_softmax_np[i],
                                        'labels': targets_np[i]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--channels', type=list, default=list(range(41)))
    parser.add_argument('--image-size', type=int, default=240)

    parser.add_argument('--data-path', type=str,
                        default=r"./data")
    parser.add_argument('--save-path', type=str,
                        default=r"./result/Segmentation")
    parser.add_argument('--model_name', type=str, default="MixBranchNet")
    parser.add_argument('--weights', type=str, default=r'./weights/Segment.pth',
                        help='initial weights path')
    parser.add_argument('--drop_rate', type=float, default=0.)

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--random_seed', default=1)
    parser.add_argument('--torch_seed', default=1)

    opt = parser.parse_args()

    main_test(opt)
