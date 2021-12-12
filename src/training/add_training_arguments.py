from argparse import ArgumentParser


def add_training_arguments(parser: ArgumentParser):
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--n_channels', help='number of channels of images', type=int, default=3)
    parser.add_argument('--hidden_size', help='size of channels of hidden representation', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--wandb_login', help='login for wandb to log process', type=str, default=None)
    parser.add_argument('--save_path', help='path to save model', type=str, default=None)
    parser.add_argument('--seed', help='fix random seed', type=int, default=0)
    return parser
