import pandas as pd
import os
import sys
import shutil
import argparse
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim
import torch
from torch import rand as random
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.optim import *
from torchvision.transforms.functional import rgb_to_grayscale
from piqa import SSIM


print(torch.__version__)
print(torchvision.__version__)
sys.stdout.flush()

import utils
from VAE.models import VQVAE_BN as VQVAE
from data_loader import CTDataset

matplotlib.use('Agg')
torch.manual_seed(0)
header= '../'
sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised training of variational autoencoder on chest CT")
    parser.add_argument(
        "--do_train",
        type=str,
        default="true",
    )

    parser.add_argument(
        "--n_iters",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mask_frac",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_channels",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="256,512",
    )

    parser.add_argument(
        "--drop_rates",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    
    parser.add_argument(
        "--slice_gap",
        type=int,
        default=0, #0 for masked region prediction, set to 1 or 3 or 5 for next slice prediction
    )

    parser.add_argument(
        "--reg_weight_mse",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--reg_weight_mask_mse",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--reg_weight_qloss",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--reg_weight_ssim",
        type=float,
        default=-0.1,
    )

    parser.add_argument(
        "--device_ids",
        type=str,
        default="0,1",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None, #only provide if model needs to be started from some checkpoint
    )

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv  file containing the validation data."
    )

    args = parser.parse_args()

    return args

def mask_loss(data0, data1, data2, data3, mask_size):
    l = 0
    data2 = data2.squeeze()
    data3 = data3.squeeze()
    for i in range(data0.shape[0]):
        r = int(data2[i].item())
        c = int(data3[i].item())
        
        l+= F.mse_loss(data0.squeeze()[i, r:r+mask_size, c:c+mask_size], data1.squeeze()[i, r:r+mask_size, c:c+mask_size])
    return l

def model_on_batch(data, model, device, mask_size, compute_loss=False):
    data[0] = Variable(data[0][0], requires_grad=True).to(device)
    data[1] = Variable(data[1][0], requires_grad=True).to(device)
    R = Variable(data[2][0], requires_grad=True).to(device)
    C = Variable(data[3][0], requires_grad=True).to(device)
    Y = model(data[0])
    
    Y[0] = rgb_to_grayscale(Y[0])
    G = rgb_to_grayscale(data[1])  
    X = Y[0].clone()
    X2 = X.view(X.shape[0],X.shape[1],-1)
    X2 -= X2.min(2, keepdim=True)[0]
    X2 /= X2.max(2, keepdim=True)[0]
    X2 = X2.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    ls1 = None
    ls2 = None
    ls3 = None
    ls4 = None
    if compute_loss:
        ls1 = F.mse_loss(Y[0], G)    
        ls2 = mask_loss(G, Y[0], R, C, mask_size)
        ls3 = torch.mean(Y[1]) #qloss
        ls4 = ssim(X2, G)

    return Y, G, X2, ls1, ls2, ls3, ls4


def save_visualizations_per_batch(fig_dir, epoch_no, batch_no, inp, gt, pred, pred_norm):
    save_image(inp, fig_dir+'train_mini_batch_'+str(epoch_no+batch_no)+'_input.jpg')
    save_image(gt, fig_dir+'train_mini_batch_'+str(epoch_no+batch_no)+'_output.jpg')
    save_image(pred, fig_dir+'train_mini_batch_'+str(epoch_no+batch_no)+'_pred.jpg')
    save_image(pred_norm, fig_dir+'train_mini_batch_'+str(epoch_no+batch_no)+'_pred_norm.jpg')
    diff_img = abs(pred-gt)
    save_image(diff_img, fig_dir+'train_mini_batch_'+str(epoch_no+batch_no)+'_diff.jpg')


def main():
    args = parse_args()

    do_train=True if args.do_train=='true' else False
    n_iters = args.n_iters
    mask_frac = args.mask_frac  ###############masking fraction , 5, 10, 20, 33 tested
    n_channels=args.n_channels
    image_size = args.image_size
    mask_size = int(image_size*(mask_frac/100))
    depth=args.depth
    hidden_dims = [int(args.hidden_dims.splot(',')[0].strip()), int(args.hidden_dims.splot(',')[1].strip())]
    drop_rates = args.drop_rates
    embedding_dim = args.embeddim_dim
    num_embeddings = args.num_embeddings
    batch_size = args.batchg_size
    volumes_per_batch = batch_size//2
    num_workers=volumes_per_batch
    slice_gap = args.slice_gap
    reg_weight_mse = args.reg_weight_mse
    reg_weight_mask_mse = args.reg_weight_mask_mse
    reg_weight_qloss = args.reg_weight_qloss
    reg_weight_ssim = args.reg_weight_ssim


    model_name = 'FoundationChestCT'
    save_dir = header+'Models/'
    fig_dir = header+'Figures/'################### path to directory to save input/output figureds after evry 10 batches
    if os.path.exists(fig_dir)==False:
        os.mkdir(fig_dir)
    device_ids = [int(args.device_ids.splot(',')[0].strip()), int(args.device_ids.splot(',')[1].strip())] ################### device ids are available gpu ids
    model_path=args.model_path


    metric_name = 'loss'
    lr = 1e-3
    maximize_metric=False
    patience = 10
    early_stop=False
    prev_val_loss = 1e10
    offset = 0

        
    # DATA
    df_train = pd.read_csv(args.train_file)
    df_val = pd.read_csv(args.validation_file)
    datagen_train = CTDataset(df =  df_train.copy(), volumes_per_batch=volumes_per_batch, batch_size_int=batch_size, slice_gap=slice_gap, mask_fraction=mask_frac, image_size = image_size, n_channels=n_channels, random_vol=False) 
    datagen_val = CTDataset(df =  df_val.copy(), volumes_per_batch=volumes_per_batch, batch_size_int=batch_size, slice_gap=slice_gap, mask_fraction=mask_frac, image_size = image_size, n_channels=n_channels, random_vol=False) 
    train_loader = DataLoader(dataset=datagen_train, shuffle=True, batch_size=1, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=datagen_val, shuffle=True, batch_size=1, num_workers=num_workers, pin_memory=True)

    print('loader created')
    sys.stdout.flush()

    model = VQVAE(in_channels= n_channels,
                    embedding_dim= embedding_dim,
                    num_embeddings =num_embeddings, 
                    hidden_dims= hidden_dims, 
                    drop_rates = drop_rates,
                    img_size= image_size, 
                    depth=depth)
    print('model created')
    sys.stdout.flush()


    save_dir = utils.get_save_dir(
        save_dir, training=True if do_train else False
    )
    
    df_loss = pd.DataFrame(columns=['train-mse-loss', 'train-qloss-sum', 'train-qloss-mean', 'train-loss', 
                                        'val-mse-loss', 'val-qloss-sum', 'val-qloss-mean', 'val-loss'])
        
    logger = utils.get_logger(save_dir, "vqvae_1ch")
    logger.propagate = False

    logger.info(' do_train: {}, n_iter: {}, batch_size: {}, volumes_per_batch:{}, slice_gap: {}, image_size: {}, depth: {}, hidden_dims:{}, drop_rates:{}, embedding_dim: {}, num_embeddings:{},  reg weight mse: {}, reg weight mask mse: {},  reg weight_qloos: {}, reg_weight_ssim:{}, mask_size: {}, model_path:{}, model: {}'.format(do_train, n_iters, batch_size,  volumes_per_batch, slice_gap, image_size, depth, hidden_dims, drop_rates, embedding_dim, num_embeddings,  reg_weight_mse, reg_weight_mask_mse, reg_weight_qloss, reg_weight_ssim,  mask_frac, model_path,  model))
    logger.info('saving to {}'.format(save_dir))
    sys.stdout.flush()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    no_of_params = count_parameters(model)
    logger.info('Number of model parameters {}'.format(no_of_params))

    logger.info('saving figures to {}'.format(fig_dir))
    sys.stdout.flush()
    ## checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=save_dir,
        metric_name=metric_name,
        maximize_metric=maximize_metric,
        log=logger,
    )



    optimizer = torch.optim.RAdam(model.parameters(),lr=lr)
    current_loss = 0
    all_losses = []

    cuda = torch.cuda.is_available()
    if cuda:
        model = nn.DataParallel(model, device_ids=device_ids)   
        device = torch.device("cuda:"+str(device_ids[0]) if torch.cuda.is_available() else "cpu")
        ssim = SSIM(n_channels=1).cuda("cuda:"+str(device_ids[0]))
        print(cuda, device)
        sys.stdout.flush()
        
        model = model.to(device)

    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    itr = 0
    prev_loss = 1e10
    prev_mse_loss = 1e10
    prev_mask_mse_loss = 1e10
    prev_ssim = 1e-10


    if do_train:
        while (itr != n_iters) and (not early_stop):
            current_loss=0
            model.train()
            #start = time.time()
            for j, data in enumerate(train_loader):
                
                Y, G, X2, ls1, ls2, ls3, ls4 = model_on_batch(data, model, mask_size, device, compute_loss=True)
                
                loss = (reg_weight_mse*ls1)+(reg_weight_mask_mse*ls2)+(reg_weight_qloss*ls3)+(reg_weight_ssim*ls4)
                
                df_loss.at[j+offset, 'train-mse-loss'] = ls1.item()
                df_loss.at[j+offset, 'train-mask-mse-loss'] = ls2.item()
                df_loss.at[j+offset, 'train-qloss-mean'] = ls3.item()
                df_loss.at[j+offset, 'train-ssim'] = ls4.item() 
                df_loss.at[j+offset, 'train-loss'] = loss.item()   
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                
                current_loss=loss.item()
                train_losses.append(current_loss)
                logger.info('##########################################TRAIN: {}, loss: {}'.format(j+offset, current_loss))
                
                if j%10==0:
                    save_visualizations_per_batch(fig_dir, j, offset, data[0], G, Y[0], X2)
                    
                    ckpt_dict = {
                        'epoch': itr,
                        'model_state': model.state_dict(),
                        'model':model,
                        'optimizer_state': optimizer.state_dict()
                    }

                    checkpoint_path = os.path.join(save_dir, 'last.pth.tar')
                    torch.save(ckpt_dict, checkpoint_path)
                    
                    if current_loss<prev_loss:
                        prev_loss = current_loss
                        best_path = os.path.join(save_dir, 'best.pth.tar')
                        shutil.copy(checkpoint_path, best_path)
                        logger.info('New best checkpoint at mini batch {}...'.format(j))
                        
                    if ls1.item()<prev_mse_loss:
                        prev_mse_loss = ls1.item()
                        best_path = os.path.join(save_dir, 'best_mse.pth.tar')
                        shutil.copy(checkpoint_path, best_path)
                        logger.info('New MSE best checkpoint at mini batch {}...'.format(j))
                        
                    if ls2.item()<prev_mask_mse_loss:
                        prev_mse_loss = ls1.item()
                        best_path = os.path.join(save_dir, 'best_mask_mse.pth.tar')
                        shutil.copy(checkpoint_path, best_path)
                        logger.info('New mask MSE best checkpoint at mini batch {}...'.format(j))
                        
                    if ls4.item()>prev_ssim:
                        prev_ssim = ls4.item()
                        best_path = os.path.join(save_dir, 'best_ssim.pth.tar')
                        shutil.copy(checkpoint_path, best_path)
                        logger.info('New SSIM best checkpoint at mini batch {}...'.format(j))
                        
                    
                    
                    model.eval()
                    with torch.no_grad():

                        data = next(iter(val_loader))
                        Y, G, X2, ls1, ls2, ls3, ls4 = model_on_batch(data, model, mask_size, device, compute_loss=True)
                        
                        loss = (reg_weight_mse*ls1)+(reg_weight_mask_mse*ls2)+(reg_weight_qloss*ls3)+(reg_weight_ssim*ls4)

                        df_loss.at[j+offset, 'val-mse-loss'] = ls1.item()
                        df_loss.at[j+offset, 'val-mask-mse-loss'] = ls2.item()
                        df_loss.at[j+offset, 'val-qloss-mean'] = ls3.item()
                        df_loss.at[j+offset, 'val-loss'] = loss.item()
                        df_loss.at[j+offset, 'val-ssim'] = ls4.item() 
                        
                        
                        loss = loss.item()
                        if loss>10:
                            loss=10
                        val_losses.append(loss)
                        logger.info('#################################################VAL: {}, loss: {}'.format(j+offset, loss))
                        
                        if j%100==0:#print val input output once in a while
                            save_visualizations_per_batch(fig_dir, j, offset, data[0], G, Y[0], X2)
                    df_loss.to_csv(os.path.join(save_dir, "loss_"+str(j+offset)+".csv"))
                    model.train()
                    
                if j%500==0 and j>0:    
                    matplotlib.use('Agg')
                    plt.figure(figsize=(10,8))
                    plt.title('Loss')
                    plt.plot(train_losses[0::10], label='train')
                    plt.plot(val_losses, label='val')
                    plt.legend(fontsize=30)
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, "graph_"+str(j+offset)+".png"))
                sys.stdout.flush()
                
            offset=j+1   
            itr+=1  

    
if __name__ == "__main__":
    main()
