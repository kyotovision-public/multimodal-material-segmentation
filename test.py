import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator



LABEL_COLORS_NEW_JP = {
    "#2ca02c" : "アスファルト",      #0
    "#1f77b4" : "コンクリート",     #1
    "#ff7f0e" : "金属",        #2
    "#d62728" : "白線", #3
    "#8c564b" : "布",#4
    "#7f7f7f" : "ガラス",        #5
    "#bcbd22" : "セメント",      #6
    "#ff9896" : "プラスチック",      #7
    "#17becf" : "ゴム",#8
    "#aec7e8" : "砂・土",         #9
    "#c49c94" : "砂利",       #10
    "#c5b0d5" : "陶器",      #11
    "#f7b6d2" : "石",  #12
    "#c7c7c7" : "レンガ",        #13
    "#dbdb8d" : "草",        #14
    "#9edae5" : "木",         #15
    "#393b79" : "葉",         #16
    "#6b6ecf" : "水",        #17
    "#9c9ede" : "人体",   #18
    "#637939" : "空"}          #19

LABEL_COLORS_NEW_EN = {
    "#2ca02c" : "asphalt",      #0
    "#1f77b4" : "concrete",     #1
    "#ff7f0e" : "metal",        #2
    "#d62728" : "road marking", #3
    "#8c564b" : "fabric, leather",#4
    "#7f7f7f" : "glass",        #5
    "#bcbd22" : "plaster",      #6
    "#ff9896" : "plastic",      #7
    "#17becf" : "rubber",#8
    "#aec7e8" : "sand",         #9
    "#c49c94" : "gravel",       #10
    "#c5b0d5" : "ceramic",      #11
    "#f7b6d2" : "cobblestone",  #12
    "#c7c7c7" : "brick",        #13
    "#dbdb8d" : "grass",        #14
    "#9edae5" : "wood",         #15
    "#393b79" : "leaf",         #16
    "#6b6ecf" : "water",        #17
    "#9c9ede" : "human body",   #18
    "#637939" : "sky"}          #19


        

        
       
        
class TesterMultimodal(object):
    def __init__(self, args):
        self.args = args

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(f'{os.path.dirname(args.pth_path)}/test')
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        input_dim = 3
        
        model = DeepLabMultiInput(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn,
                        input_dim=input_dim,
                        ratio=args.ratio,
                        pretrained=args.use_pretrained_resnet)
        
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        
        # Define Optimizer
        optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,lr=args.lr,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, ignore_index=0).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Load model parameters
        checkpoint = torch.load(args.pth_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print(checkpoint['epoch'])
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def test(self, epoch=0):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        scaler = torch.cuda.amp.GradScaler()
        image_all = None
        target_all = None
        output_all = None
        for i, sample in enumerate(tbar):
            image, target, aolp, dolp, nir, nir_mask, uvmap,mask = \
                sample['image'], sample['label'], sample['aolp'], sample['dolp'], sample['nir'], sample['nir_mask'], sample['uvmap'],sample['mask']
            if self.args.cuda:
                image, target, aolp, dolp, nir, nir_mask, uvmap,mask = image.cuda(), target.cuda(), aolp.cuda(), dolp.cuda(), nir.cuda(), nir_mask.cuda(), uvmap.cuda(),mask.cuda()
            aolp = aolp if self.args.use_aolp else None
            dolp = dolp if self.args.use_dolp else None
            nir  = nir  if self.args.use_nir else None
            nir_mask = nir_mask  if self.args.use_nir else None

            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output= self.model(image, aolp, dolp, nir, mask)
                loss = self.criterion(output, target, nir_mask)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            pred = output.data.cpu().numpy()
            target_ = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            if image_all is None:
                image_all  =  image.cpu().clone()
                target_all = target.cpu().clone()
                output_all = output.cpu().clone()
            else:
                image_all  = torch.cat(( image_all, image.cpu().clone()),dim=0)
                target_all = torch.cat((target_all,target.cpu().clone()),dim=0)
                output_all = torch.cat((output_all,output.cpu().clone()),dim=0)
                
            # Add batch sample into evaluator
            self.evaluator.add_batch(target_, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        confusion_matrix = self.evaluator.confusion_matrix
        np.save(f'{os.path.dirname(args.pth_path)}/test/confusion_matrix.npy',confusion_matrix)

        self.writer.add_scalar('test/mIoU', mIoU, epoch)
        self.writer.add_scalar('test/Acc', Acc, epoch)
        self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        self.summary.visualize_test_image(self.writer, self.args.dataset, image_all, target_all, output_all, 0)
        
        print('Test:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        #print('Loss: %.3f' % test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnet_adv', 'xception_adv','resnet_condconv'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'kitti', 'kitti_advanced', 'kitti_advanced_manta', 'handmade_dataset', 'handmade_dataset_stereo', 'multimodal_dataset'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'original'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=True,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--ratio', type=int, default=None, metavar='N',
                        help='number of ratio in RGFSConv (default: 1)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    # propagation and positional encoding option
    parser.add_argument('--propagation', type=int, default=0,
                        help='image propagation length (default: 0)')
    parser.add_argument('--positional-encoding', action='store_true', default=False,
                        help='use positional encoding')
    parser.add_argument('--use-aolp', action='store_true', default=False,
                        help='use aolp')
    parser.add_argument('--use-dolp', action='store_true', default=False,
                        help='use dolp')
    parser.add_argument('--use-nir', action='store_true', default=False,
                        help='use nir')
    parser.add_argument('--use-pretrained-resnet', action='store_true', default=False,
                        help='use pretrained resnet101')
    parser.add_argument('--list-folder', type=str, default='list_folder1')
    parser.add_argument('--is-multimodal', action='store_true', default=False,
                        help='use multihead architecture')
    parser.add_argument('--pth-path', type=str, default=None,
                        help='set the pth file path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'kitti': 50,
            'kitti_advanced': 50
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'kitti' : 0.01,
            'kitti_advanced' : 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    # input('Check arguments! Press Enter...')
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.is_multimodal:
        print("USE Multimodal Model")
        tester = TesterMultimodal(args)
    else:
        tester = TesterAdv(args)
    print('Starting Epoch:', tester.args.start_epoch)
    print('Total Epoches:', tester.args.epochs)
    tester.test()
    tester.writer.close()
    print(args)