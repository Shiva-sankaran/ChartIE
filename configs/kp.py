import argparse
parser_kp = argparse.ArgumentParser()

parser_kp.add_argument('--batch_size', default=42, type=int)
parser_kp.add_argument('--patch_size', default=16, type=int)

parser_kp.add_argument('--position_embedding', default='enc_xcit',
					type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
										'learned_cls', 'learned_nocls', 'none'),
					help="Type of positional embedding to use on top of the image features")
parser_kp.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
					help="Activation function used for the transformer decoder")

parser_kp.add_argument('--input_size', nargs="+", default=[288, 384], type=int,
					help="Input image size. Default is %(default)s")
parser_kp.add_argument('--hidden_dim', default=384, type=int,
					help="Size of the embeddings for the DETR transformer")

parser_kp.add_argument('--vit_dim', default=384, type=int,
					help="Output token dimension of the VIT")
parser_kp.add_argument('--vit_weights', type=str, default="https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth",
					help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")
parser_kp.add_argument('--vit_dropout', default=0., type=float,
					help="Dropout applied in the vit backbone")
parser_kp.add_argument('--dec_layers', default=6, type=int,
					help="Number of decoding layers in the transformer")
parser_kp.add_argument('--dim_feedforward', default=1536, type=int,
					 help="Intermediate size of the feedforward layers in the transformer blocks")
parser_kp.add_argument('--dropout', default=0., type=float,
					help="Dropout applied in the transformer")
parser_kp.add_argument('--nheads', default=8, type=int,
					help="Number of attention heads inside the transformer's attentions")
parser_kp.add_argument('--num_queries', default=64, type=int,
					help="Number of query slots")
parser_kp.add_argument('--pre_norm', action='store_true')
parser_kp.add_argument('--data_path', default="/home/vp.shivasan/data/data/ChartOCR_lines/line/images", type=str)

parser_kp.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
														(default from simple baselines is %(default)s)")

parser_kp.add_argument('--num_workers', default=16, type=int)
