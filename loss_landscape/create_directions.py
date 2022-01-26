"""
Code to compute directions
    This code computes normalized random directions and PCA directions.
"""

import argparse
import logging
import os
import sys

import dill
import numpy
import torch

from utils.evaluations import get_PCA_directions
from utils.nn_manipulation import count_params, create_normalized_random_direction
from utils.reproducibility import set_seed
from utils.resnet import get_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument(
        "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--result_folder", "-r", required=True)

    # model related arguments
    parser.add_argument(
        "--statefile", "-s", required=False, default=None,
        help="required to compute random directions"
    )
    parser.add_argument(
        "--statefile_folder", required=False, default=None,
        help="required for computing PCA directions"
    )
    parser.add_argument("--skip_bn_bias", action="store_true")
    parser.add_argument(
        "--model", required=True, choices=["resnet20", "resnet32", "resnet44", "resnet56"]
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)

    parser.add_argument("--batch_size", required=False, type=int, default=128)
    parser.add_argument(
        "--direction_file", required=True, type=str, help="file name to store directions"
    )
    parser.add_argument(
        "--direction_style", required=True, type=str,
        choices=["random", "pca", "frequent_directions"]
    )

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    set_seed(args.seed)
    if os.path.exists(f"{args.result_folder}/{args.direction_file}"):
        logger.error("direction file exists, so we will exit")
        sys.exit()

    if args.direction_style == "random":
        # get model
        model = get_resnet(args.model)(
            num_classes=10, remove_skip_connections=args.remove_skip_connections
        )
        model.to(args.device)
        total_params = count_params(model)
        logger.info(f"using {args.model} with {total_params} parameters")

        # we need to load actual weights to get the magnitudes for normalization
        logger.info(f"Loading model from {args.statefile}")
        state_dict = torch.load(args.statefile, pickle_module=dill, map_location=args.device)
        model.load_state_dict(state_dict)

        # create "filter" normalized random direction if nothing is passed
        direction1 = create_normalized_random_direction(model, skip_bn_bias=True)
        direction2 = create_normalized_random_direction(model, skip_bn_bias=True)


        def flatten_direction(direction):
            for i in range(len(direction)):
                direction[i] = torch.flatten(direction[i])
            return torch.cat(direction, dim=0)


        direction1 = flatten_direction(direction1).cpu().data.numpy()
        direction2 = flatten_direction(direction2).cpu().data.numpy()
        numpy.savez(
            f"{args.result_folder}/{args.direction_file}", direction2=direction2,
            direction1=direction1
        )

    if args.direction_style == "pca":
        # get model
        model = get_resnet(args.model)(num_classes=10)
        model.to("cpu")
        # because here we will be mainly moving data so using cpu should be a better idea
        logger.info(f"using {args.model} with {count_params(model)} parameters")

        state_files = [f"{args.statefile_folder}/init_model.pt"]
        i = 1
        while os.path.exists(f"{args.statefile_folder}/{i}_model.pt"):
            state_files.append(f"{args.statefile_folder}/{i}_model.pt")
            i += 1
        logger.info(f"Found {len(state_files)} models")

        direction1, direction2, ex_var = get_PCA_directions(model, state_files, skip_bn_bias=True)
        numpy.savez(
            f"{args.result_folder}/{args.direction_file}", explained_variance=ex_var,
            direction2=direction2, direction1=direction1
        )

    if args.direction_style == "frequent_directions":
        logger.info("See train.py to generate this")
