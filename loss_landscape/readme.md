This is the starter code for homework 1 (visualizing loss landscape) and is
heavily based on https://github.com/tomgoldstein/loss-landscape.

## Setup

- See `install.sh` or `requirements.txt` for required packages.
- Tested with python3.8

## How to use?

- **Training the models**: Use `train.py` to train the model. Currently, you can
  train resnets on CIFAR 10. The usage is below.
    - Notes:
        - Use `mode` to provide if you want to test, train or both.
        - Use `statefile` to init with some set of weights (or load a pretrained
          model).
        - `remove_skip_connections` will eliminate all skip connections.
        - `skip_bn_bias` removes batch norm and bias from list of parameters
          when flattening the params/grads for projection or computing
          directions. This was used for frequent directions algorithm which is
          used to compute streaming SVD of all the gradients. Li et al. (2018)
          do not consider bias and batch norm parameters in their work.

```
usage: train.py [-h] [-D] [--seed SEED] 
                [--device DEVICE] 
                --result_folder RESULT_FOLDER 
                [--mode {test,train} [{test,train} ...]] 
                --statefile STATEFILE 
                --model {resnet20,resnet32,resnet44,resnet56} 
                [--remove_skip_connections]
                [--batch_size BATCH_SIZE] 
                [--save_strategy {epoch,init} [{epoch,init} ...]] 
                [--skip_bn_bias]
                
example:  
python train.py \
      --result_folder "results/resnet56_skip_bn_bias_remove_skip_connections/" \
      --device cuda:3 --model resnet56 \
      --skip_bn_bias -D --remove_skip_connections 
```

- **Creating direction for projection**: We need two directions (vectors) on
  which we project the project weights for different visualizations. We can
  create directions in different ways and we provide following ways:
    1. Random directions
    2. Principle vector of {w_final-w_i} where i are models saved at end of each
       epoch.
    3. (Approximate) SVD of gradients during training (computed using frequent
       direction algorithm)
        1. Gradient during all training
        2. last 10 epoch
        3. last epoch

  3 are computed during training, for 1 and 2 use `create_directions.py`

```commandline
usage example:

- creating random directions with filter normalization (the checkpoint weights are used for normalization).

python create_directions.py --statefile results/resnet20_skip_bn_bias/ckpt/200_model.pt \
    -r results/resnet20_skip_bn_bias/ --skip_bn_bias --direction_file random_directions.npz \
    --direction_style "random"  --model resnet20

- for pca direction (the statefile folder is folder of all checkpoints)
python create_directions.py \
    --statefile_folder results/resnet20_skip_bn_bias/ckpt/ \
    -r results/resnet20_skip_bn_bias --skip_bn_bias \
    --direction_file pca_directions.npz --direction_style "pca" \
    --model resnet20

```

- **Computing Optimization Trajectories**:

```commandline
python compute_trajectory.py -r results/resnet20_skip_bn_bias_remove_skip_connections/trajectories \
  --direction_file results/resnet20_skip_bn_bias_remove_skip_connections/pca_directions.npz \
  --projection_file pca_dir_proj.npz --model resnet20  --remove_skip_connections \
  -s results/resnet20_skip_bn_bias_remove_skip_connections/ckpt --skip_bn_bias \
 ```

This takes a folder of checkpoints (-s argument) and computes the projection of
$w_i-w_final$ on the direction vectors. The results are saved to the projection
file.

- **Computing loss landscapes of final models**

```commandline
python compute_loss_surface.py \
    --result_folder results/resnet20_skip_bn_bias_remove_skip_connections/loss_surface/  \
    -s results/resnet20_skip_bn_bias_remove_skip_connections/ckpt/200_model.pt \
    --batch_size 1000 --skip_bn_bias \
    --model resnet20 --remove_skip_connections \
    --direction_file results/resnet20_skip_bn_bias_remove_skip_connections/pca_directions.npz \
    --surface_file pca_dir_loss_surface.npz --device cuda:0 \
    --xcoords 51:-10:40 --ycoords 51:-10:40  
```

- **Plotting results**:
    - You can pass either trajectory file or surface file or both in the command
      below.

```
python plot.py --result_folder figures/resnet56/ \
    --trajectory_file results/resnet56_skip_bn_bias/trajectories/pca_dir_proj.npz \
    --surface_file results/resnet56_skip_bn_bias/loss_surface/pca_dir_loss_surface.npz \
    --plot_prefix resnet56_pca_dir
```

Note: The code should be executable with loss-landscape as the root folder. 