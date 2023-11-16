import torch
from utils.F1_Score import f1_score


def evaluate(net, dataloader, device,mulout=False):
    net.eval()
    n = len(dataloader)
    F1=0
    se=0
    sp=0
    acc=0
    for batch in dataloader:
        image,mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        with torch.no_grad():
            if mulout:
                pred=net(image)[-1]
            else:
                pred = net(image)

            if net.n_classes == 1:
                pred = (torch.sigmoid(pred) > 0.5).float()
                pred = pred.squeeze(1)
            else:
                pred = torch.argmax(pred, dim=1)
            out=f1_score(pred,mask_true)
            F1+=out[0]
            se+=out[1]
            sp += out[2]
            acc += out[3]
    net.train()
    return F1/n,se/n,sp/n,acc/n