import argparse
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import RandomResizedCrop
import torch.nn as nn

from eval_f1 import evaluate
from mynet import UNet
from utils.data_loading import BasicDataset


def get_save_path():
    if not os.path.exists('log'):
        os.mkdir('log')
    n=1
    while True:
        path='log/train_'+str(n)
        if os.path.exists(path):
            n+=1
            continue
        os.mkdir(path)
        break
    return path


dir_train = './DRIVE/train/'
dir_test = './DRIVE/test/'


def train_model(
        model,
        device,
        epochs: int,
        batch_size: int,
        learning_rate: float,

        weight_decay: float = 0,
        gradient_clipping: float = 1.0,
):

    train_set = BasicDataset(dir_train, mask_suffix='',is_train=True,W=192,H=192)
    test_set= BasicDataset(dir_test,mask_suffix='',is_train=False)

    n_train = len(train_set)

    train_loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **test_loader_args)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)

    criterion = nn.BCEWithLogitsLoss().to(device)

    max_test_F1=0
    loss_lr_note=[]
    cropresize=RandomResizedCrop(size=128,scale=(1./9.,1.0)).to(device)


    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                img_mask = torch.cat([images, true_masks.unsqueeze(1)], dim=1)
                img_mask = cropresize(img_mask)
                images = img_mask[:, 0:3, ...]
                true_masks = img_mask[:, -1, ...]
                true_masks = (true_masks > 0.5)

                masks_pred = model(images)

                loss=criterion(masks_pred.squeeze(1), true_masks.float())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])

                pbar.set_postfix(**{'loss': loss.item(),'lr': optimizer.state_dict()['param_groups'][0]['lr']})
                loss_lr_note.append('loss:'+str(loss.item())+' lr:'+str(optimizer.state_dict()['param_groups'][0]['lr']))

        test_F1, test_se, test_sp, test_acc = evaluate(model, test_loader, device, mulout=False)
        print('test_F1', test_F1)
        print('test_se', test_se)
        print('test_sp', test_sp)
        print('test_acc', test_acc)
        if test_F1.item() > max_test_F1:
            max_test_F1 = test_F1.item()
            torch.save(model.state_dict(), save_path + '/best.pth')

        with open(save_path + "/test.txt", "a") as f:
            f.write('f1:' + str(test_F1.item()) + ' se:' + str(test_se.item()) + ' sp:' + str(test_sp.item()) + ' acc:' + str(test_acc.item()) + '\n')

        print('max_F1',max_test_F1)
        scheduler.step()

        if len(loss_lr_note):
            with open(save_path+"/loss_lr.txt", "a") as f:
                f.write('\n'.join(loss_lr_note)+'\n')
            loss_lr_note=[]

        if epoch>90:
            torch.save(model.state_dict(), save_path + '/'+str(epoch)+'.pth')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    paths = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n = 1
    for i in range(n):
        torch.cuda.empty_cache()
        print('--------------------------')
        print('--------------------------')
        print('--------------------------')
        save_path = get_save_path()
        paths.append(save_path)
        print('save_path',save_path)

        model = UNet(n_channels=3, n_classes=args.classes)

        if args.load:
            state_dict = torch.load(args.load, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device=device)
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
        )
        print(paths)
        print('save_path', save_path)
