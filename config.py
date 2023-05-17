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
                        default='./dataset/train')
    parser.add_argument('--test_data_path',
                        default='./dataset/test')

    # training
    parser.add_argument('--resume', default=None, type=str,
                        help='pretranined model path')
    parser.add_argument('--N_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--early-stop', default=6, type=int,
                        help='early stopping')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--val_on_test', default=False, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model)load trained model to continue train')

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args