
import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from optimizer import build_optimizer
from logger import create_logger
from utils import *
from evaluation import *
from selections import *
#from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dtaidistance import dtw






try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True,
                        help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    train_writer = SummaryWriter('../runs_'+config.MODEL.NAME+'/swin')
    return args, config,train_writer


def main(config,train_writer):

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config).to(config.TRAIN.CUDA)
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    #lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    lr_scheduler=None

    criterion_triplet = nn.TripletMarginLoss(margin=config.TRAIN.MARGIN, p=2)
    criterion_mse = nn.MSELoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model, optimizer, lr_scheduler, logger)
        if config.EVAL_MODE:
            return

    optimizer.param_groups[0]['lr']=config.TRAIN.BASE_LR

    dataset_train, dataset_val,dataset_test,data_loader_train, data_loader_val,data_loader_test,mixup_fn=build_loader(config)
    status,name_map=None,None
    ComputeSim = ComputeSimilarityGroundTruth(
        data_folder=config.DATA.DATA_PATH+"/data",
        distance_metric='dtw'
    )
    ComputeSimVal = ComputeSimilarityGroundTruth(
        data_folder=config.DATA.DATA_PATH+"/data",
        distance_metric='dtw'
    )
    ComputeSimTest = ComputeSimilarityGroundTruth(
        data_folder=config.DATA.DATA_PATH+"/data",
        distance_metric='dtw'
    )
    if config.TRAIN.TRIPLET_ENABLE:

        trainNames = os.listdir(config.DATA.DATA_PATH+'/train')
        DistMatrix = ComputeSim.build_dist_matrix(
            trainNames, load=config.TRAIN.LOAD_MAT, path=config.DATA.DATA_PATH+'/saved/')
        NormDM = ComputeSim.normalize_simple(DistMatrix)
        

    valNames=os.listdir(config.DATA.DATA_PATH+'/val')
    testNames=os.listdir(config.DATA.DATA_PATH+'/test')

    DMVal = ComputeSimVal.build_dist_matrix(
        valNames, load=config.TRAIN.LOAD_MAT, path=config.DATA.DATA_PATH+'/saved_val/',interpolation=False)
    DMTest = ComputeSimTest.build_dist_matrix(
        testNames, load=config.TRAIN.LOAD_MAT, path=config.DATA.DATA_PATH+'/saved_test/',interpolation=False)


    logger.info("Start training")
    start_time = time.time()

    bound_idx=-1
    for idx,epoch in enumerate(config.TRAIN.SWITCH):
        if config.TRAIN.START_EPOCH>epoch:
            bound_idx=idx

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        saved_ckpt=False
        if ((bound_idx<len(config.TRAIN.SWITCH)-1 and epoch==config.TRAIN.SWITCH[bound_idx+1]) or epoch==config.TRAIN.START_EPOCH) \
                and config.TRAIN.TRIPLET_ENABLE:
            if (bound_idx<len(config.TRAIN.SWITCH)-1 and epoch==config.TRAIN.SWITCH[bound_idx+1]):
                bound_idx+=1
            status,name_map=calc_examples(config.TRAIN.PBOUNDS[bound_idx],config.TRAIN.NBOUNDS[bound_idx],NormDM,ComputeSim)
            saved_ckpt=True

        if dist.get_rank()==0 and (epoch % config.SAVE_FREQ == config.SAVE_FREQ-1 or saved_ckpt)and epoch!=config.TRAIN.START_EPOCH:
            save_checkpoint(config, epoch-1, model,max_accuracy, optimizer, lr_scheduler, logger)

        if config.TRAIN.TRIPLET_ENABLE:
            train_one_epoch_triplet(config, model, criterion_triplet, criterion_mse, dataset_train,
                                    data_loader_train, optimizer, epoch, lr_scheduler, status, name_map,NormDM,train_writer)
        else:
            train_one_epoch(config, model, criterion_triplet, criterion_mse, dataset_train,
                            data_loader_train, optimizer, epoch, lr_scheduler, status, name_map,train_writer) 

        if epoch%5==4:
        #if True:
            vec, records = calcEmbed(config, data_loader_val, model)
            vec = np.array(vec)
            vec.resize(vec.shape[0], config.MODEL.FEATURE_SIZE)

            knn = KNeighborsClassifier(n_neighbors=config.TEST.TOPK+1)
            knn.fit(vec, range(len(vec)))

            mapk, precision,hr10,hr50,r10_50,r10_100 = analysis(vec, [], DMVal, ComputeSimVal, knn, records, config.TEST.TOPK)
            train_writer.add_scalar('mapk', mapk, epoch)
            train_writer.add_scalar('precision', precision, epoch)
            train_writer.add_scalar('hr10', hr10, epoch)
            train_writer.add_scalar('hr50', hr50, epoch)
            train_writer.add_scalar('r10@50', r10_50, epoch)
            train_writer.add_scalar('r10@100', r10_100, epoch)

    if dist.get_rank() == 0:
        save_checkpoint(config, config.TRAIN.EPOCHS-1, model,max_accuracy, optimizer, lr_scheduler, logger)   

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    vec, records = calcEmbed(config, data_loader_test, model)
    vec = np.array(vec)

    knn = KNeighborsClassifier(n_neighbors=config.TEST.TOPK+1)
    knn.fit(vec, range(len(vec)))
    analysis(vec, [], DMTest, ComputeSimTest, knn, records, config.TEST.TOPK)

    pca = PCA(n_components=2)
    pca.fit(vec)
    vec_ = pca.transform(vec)
    plt.plot(vec_[:, 0], vec_[:, 1], '.', alpha=0.05)
    plt.savefig('sne.jpg')

def train_one_epoch(config, model, criterion_triplet, criterion_mse, dataset, train_loader, optimizer, epoch, lr_scheduler, status, name_map,train_writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    triplet_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (datas, names,_) in enumerate(train_loader):


        sample = datas.to(config.TRAIN.CUDA)
        codes, outputs = model(sample)
        mse_loss = criterion_mse(outputs, sample)
        triplet_loss = 0
        loss = mse_loss+triplet_loss

        if loss == torch.inf or loss == torch.nan:
            loss = 0
            mse_loss = 0
            triplet_loss = 0
            optimizer.zero_grad()

        else:
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    #lr_scheduler.step_update(epoch * num_steps + idx)
            else:
                optimizer.zero_grad()
                if config.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
            #lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        if loss != 0:
            loss_meter.update(loss.item())
        if mse_loss != 0:
            mse_meter.update(mse_loss.item())
        if triplet_loss != 0:
            triplet_meter.update(triplet_loss.item())
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or config.TRAIN.TRIPLET_ENABLE:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'mse loss {mse_meter.val:.4f} ({mse_meter.avg:.4f})\t'
                f'triplet loss {triplet_meter.val:.4f} ({triplet_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    train_writer.add_scalar('loss', loss_meter.avg, epoch)


def train_one_epoch_triplet(config, model, criterion_triplet, criterion_mse, dataset, train_loader, optimizer, epoch, lr_scheduler, status, name_map,NormDM,train_writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    triplet_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (datas, names, series) in enumerate(train_loader):


        if len(names)<config.DATA.BATCH_SIZE:
            continue

        mse_tot = 0
        triplet_tot = 0
        loss_tot = 0
        loss = 0
        hit=0
        zero=0

        
        if config.TRAIN.TRIPLET_SELECTION=='diversified':
            vec = calcVec(config, datas, model)
            vec = normalize(vec, axis=1)
            vec = torch.Tensor(vec)
            torch.cuda.empty_cache()

            samples=triplet_select_diversified(config,vec,config.TRAIN.TRIPLET_CLUSTERS,series,names,status,name_map,datas,NormDM)
            tot=candidates=len(samples)
        else:
            tot=0
            candidates=len(names)

        for i in range(candidates//config.TRAIN.TRIPLET_BATCH):
            if i%32==0 and config.TRAIN.TRIPLET_SELECTION!='diversified':
                vec = calcVec(config, datas, model)
                vec = normalize(vec, axis=1)
                vec = torch.Tensor(vec)
                torch.cuda.empty_cache()

            if config.TRAIN.TRIPLET_SELECTION=='hard':
                sample=triplet_select_hard(config,vec,names,status,name_map,datas,names[i*config.TRAIN.TRIPLET_BATCH:(i+1)*config.TRAIN.TRIPLET_BATCH])
            elif config.TRAIN.TRIPLET_SELECTION=='semi_hard':
                sample=triplet_select_semi_hard(config,vec,names,status,name_map,datas,names[i*config.TRAIN.TRIPLET_BATCH:(i+1)*config.TRAIN.TRIPLET_BATCH])
            elif config.TRAIN.TRIPLET_SELECTION=='all':
                sample=triplet_select_all(config,names,status,name_map,datas,names[i*config.TRAIN.TRIPLET_BATCH:(i+1)*config.TRAIN.TRIPLET_BATCH])
            
            if config.TRAIN.TRIPLET_SELECTION=='diversified':
                sample = torch.cat(samples[int(i*config.TRAIN.TRIPLET_BATCH):int((i+1)*config.TRAIN.TRIPLET_BATCH)], 0).to(config.TRAIN.CUDA)
            else:
                tot+=len(sample)
                if len(sample)==0:
                    continue
                sample = torch.cat(sample, 0).to(config.TRAIN.CUDA)

            codes, outputs = model(sample)

            triplet_loss=0
            #mse_loss = criterion_mse(outputs, sample)
            mse_loss=0
            for j in range(sample.shape[0]//3):
                m_loss = criterion_mse(outputs[j*3:(j+1)*3], sample[j*3:(j+1)*3])
                t_loss=criterion_triplet(codes[j*3].reshape(1,-1), codes[j*3+1].reshape(1,-1), codes[j*3+2].reshape(1,-1))
                if t_loss!=0:
                    mse_loss+=m_loss
                    triplet_loss+=t_loss
                #triplet_loss += t_loss
                if t_loss>config.TRAIN.MARGIN:
                    hit+=1
                elif t_loss==0:
                    zero+=1
            
            if triplet_loss==0:
                continue

            loss = triplet_loss.to(config.TRAIN.CUDA)*config.TRAIN.TRIPLET_WEIGHT+mse_loss
            loss/=(config.TRAIN.TRIPLET_WEIGHT+1)
            #loss=triplet_loss.to(config.TRAIN.CUDA)
                
            mse_tot += float(mse_loss)
            triplet_tot += float(triplet_loss)
            loss_tot += float(loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss=0
            mse_loss=0
            triplet_loss=0
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        tot-=zero
        loss_tot/=tot
        mse_tot/=tot
        triplet_tot/=tot
        loss_meter.update(loss_tot)
        mse_meter.update(mse_tot)
        triplet_meter.update(triplet_tot)

        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())

        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]['lr']
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        logger.info(
            f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
            f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'mse loss {mse_meter.val:.4f} ({mse_meter.avg:.4f})\t'
            f'triplet loss {triplet_meter.val:.4f} ({triplet_meter.avg:.4f})\t'
            f'grad_norm {norm_meter.val} ({norm_meter.avg})\t'
            f'mem {memory_used:.0f}MB\t'
            f'triplet hit {hit}\t'
            f'triplet zero {zero}\t'
            f'triplet get {tot}')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    train_writer.add_scalar('loss', loss_meter.avg, epoch)
    train_writer.add_scalar('mse', mse_meter.avg, epoch)
    train_writer.add_scalar('triplet', triplet_meter.avg, epoch)




if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    _, config,train_writer = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    #torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    '''config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()'''

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config,train_writer)