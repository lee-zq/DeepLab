import argparse

def global_parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments/cls_dense_net',
                        help='trained model will be saved at here')
    parser.add_argument('--name', default='default_name',
                        help='save name of experiment in args.outf directory')

    # data
    parser.add_argument('--train_data_path',
                        default='./datasets/cifar10')
    parser.add_argument('--test_data_path',
                        default='./datasets/cifar10')

    # training
    parser.add_argument('--checkpoint', default=None,
                        help='(path of trained _model)load trained model to continue train')
    parser.add_argument('--n_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=6, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args