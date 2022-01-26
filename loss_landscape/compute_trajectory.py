"""
    Code to compute 2D projection of a list of model,i.e., optimization direction along given directions
"""

import argparse
import logging
import os
import sys

import dill
import numpy
import torch

from utils.nn_manipulation import count_params, flatten_params
from utils.reproducibility import set_seed
from utils.resnet import get_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--result_folder", "-r", required=True)

    # model related arguments
    parser.add_argument("--statefile_folder", "-s", required=True)
    parser.add_argument(
        "--model", required=True, choices=["resnet20", "resnet32", "resnet44", "resnet56"]
    )
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument("--skip_bn_bias", action="store_true")

    parser.add_argument("--direction_file", required=True)
    parser.add_argument("--projection_file", required=True)

    args = parser.parse_args()

    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if os.path.exists(f"{args.result_folder}/{args.projection_file}"):
        logger.error(f"{args.projection_file} exists, so we will exit")
        sys.exit()

    set_seed(args.seed)

    # get model
    model = get_resnet(args.model)(
        num_classes=10, remove_skip_connections=args.remove_skip_connections
    )
    model.to("cpu")
    # since we will be mainly moving data so using cpu should be a better idea
    logger.info(f"using {args.model} with {count_params(model)} parameters")

    state_files = [f"{args.statefile_folder}/init_model.pt"]
    i = 1
    while os.path.exists(f"{args.statefile_folder}/{i}_model.pt"):
        state_files.append(f"{args.statefile_folder}/{i}_model.pt")
        i += 1
    logger.info(f"Found {len(state_files)} models")

    temp = numpy.load(args.direction_file)
    direction1 = torch.tensor(temp["direction1"], device="cpu").float()
    direction2 = torch.tensor(temp["direction2"], device="cpu").float()

    model.load_state_dict(torch.load(state_files[-1], pickle_module=dill, map_location="cpu"))
    total_param = count_params(model, skip_bn_bias=args.skip_bn_bias)
    w_final = flatten_params(model, total_param, skip_bn_bias=args.skip_bn_bias)

    w_diff_matrix = torch.zeros(len(state_files) - 1, total_param)
    for idx, file in enumerate(state_files[:-1]):
        model.load_state_dict(torch.load(file, pickle_module=dill, map_location="cpu"))
        w = flatten_params(model, total_param, skip_bn_bias=args.skip_bn_bias)

        diff = w - w_final
        w_diff_matrix[idx] = diff

    # do the projection
    logger.info(f"Dot product is {direction1 @ direction2}")
    if torch.isclose(direction1 @ direction2, torch.tensor(0.0)):
        logger.info("The directions are orthogonal")
        # when dx and dy are orthorgonal
        xcoords = w_diff_matrix @ direction1 / direction1.norm()
        ycoords = w_diff_matrix @ direction2 / direction2.norm()
    else:
        # w_diff (nxd)
        # A = dx2
        # X = 2xn
        # AX = w_diff.T
        # solve the least squre problem: Ax = d
        A = torch.vstack([direction1, direction2]).T  # num_param X 2
        temp = torch.linalg.lstsq(A, w_diff_matrix.T).solution  # 2
        xcoords, ycoords = temp[0], temp[1]

    # save losses and accuracies evaluations
    logger.info("Saving results")
    numpy.savez(
        f"{args.result_folder}/{args.projection_file}", xcoordinates=xcoords.cpu().data.numpy(),
        ycoordinates=ycoords.cpu().data.numpy()
    )

    logger.info(f"xrange: {xcoords.min()}, {xcoords.max()}")
    logger.info(f"yrange: {ycoords.min()}, {ycoords.max()}")
