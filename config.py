import numpy as np
import os

import datasets
import utils

from parameters import HParamSelect, ParamsDict

'''
Params definition. You can use use auxiliary classes like HParamSelect and HParamRange
to add some randomness into exact values selection process.
'''
params_defs = ParamsDict(
    batch_size=HParamSelect([128]),
    noise_size=100,
    noise_scale=0.1,
    pretrain_steps=10**2,
    steps=10**5,
    epochs=3,
    dis_lr=0.0003,
    dis_filters=HParamSelect([96]),
    dis_filters_size=HParamSelect([5]),
    gen_lr=0.0001,
    gen_filters=HParamSelect([96]),
    gen_filters_size=HParamSelect([5]),
    gen_keep_dropout=0.9,
    use_batch_norm=HParamSelect([True]),
    loss_diff_threshold=10.0,
    loss_diff_threshold_back=1.0,
    gen_scope='gen',
    dis_scope='dis',
    save_steps=np.array([i for i in range(5)]) * 5*10**4,
    save_old_steps=np.arange(20) * 10000,
    switch_model_loss_decay=0.95,
    summaries_steps=200,
    prints_steps=200,
    draw_steps=4000,
    debug=False,
    # To run on CIFAR-10, please use Cifar10Dataset
    dataset=lambda: datasets.MnistDataset(),
    model_path='',
    mode='train',
    nb_generated=4000,
    show_h_images=5,
    show_w_images=5,
    show_figsize=5)


class GANParams(ParamsDict):
    """A GANParams contains all the settings needed for GAN training."""

    def __init__(self, *args, **kwargs):
        super(GANParams, self).__init__(*args, **kwargs)
        # Parse parameters.
        self.name = 'gan_%s_Dlr%.4f_Glr%.4f_Df%d,_Dfs%d_Gf%d,_Gfs%d_Gdp%.2f_bs%d' % (
            utils.get_date(),
            self.dis_lr, self.gen_lr,
            self.dis_filters, self.dis_filters_size,
            self.gen_filters, self.gen_filters_size,
            self.gen_keep_dropout,
            self.batch_size)
        self.checkpoint_dir = 'checkpoints/%s' % self.name
        self.images_dir = 'images/%s/%s' % (self.dataset.name, self.name)
        for directory in [self.checkpoint_dir, self.images_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)