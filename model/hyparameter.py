# -- coding: utf-8 --

import argparse
class parameter(object):
    def __init__(self,parser):
        self.parser=parser

        self.parser.add_argument('--save_path', type=str, default='end_to_end/', help='save path')

        self.parser.add_argument('--target_site_id', type=int, default=0, help='city ID')
        self.parser.add_argument('--pollutant_id', type=str, default='PM2.5', help='pollutant name')
        self.parser.add_argument('--data_divide', type=float, default=1.0, help='data_divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epochs', type=int, default=50, help='epoch')
        self.parser.add_argument('--step', type=int, default=24, help='step')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='drop out')
        self.parser.add_argument('--site_num', type=int, default=47, help='total number of city')

        self.parser.add_argument('--features', type=int, default=10, help='numbers of the feature')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=48, help='input length')
        self.parser.add_argument('--output_length', type=int, default=24, help='output length')

        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

        self.parser.add_argument('--hidden_size', type=int, default=128, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=1.0, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.0, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=1.0, help='test set rate')

        self.parser.add_argument('--file_train', type=str,
                                 default='data/train.csv',
                                 help='training set file address')
        self.parser.add_argument('--file_val', type=str,
                                 default='data/val.csv',
                                 help='validate set file address')
        self.parser.add_argument('--file_test', type=str,
                                 default='data/test.csv',
                                 help='test set file address')

        self.parser.add_argument('--file_adj', type=str,
                                 default='data/adjacent.csv',
                                 help='adj file address')

        self.parser.add_argument('--file_out', type=str, default='ckpt', help='file out')

    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)