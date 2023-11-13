# default returns a list or arbitrary length depending on __init__
# default/full order:     label,[sequence/sequence+tag,[geneexp,geneloc]]
# sequence                [N,2001,4]
# label                   [N,23]
# tag                     [N,2001,1]
# geneexp                 [G]
# geneloc sparse array    [N,G]
# geo2vec                 [N,12] (shared [N,6],tissue [N,6])
import numpy as np
import copy
from torch import nn
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def evaluate(pred, y, prefix='', trim=0):
    # assume label to be [N]
    pred = np.squeeze(pred.cpu().detach().numpy())
    y = np.squeeze(y.cpu().detach().numpy())
    predint = (pred > 0).astype(int)
    if trim != 0:
        posidx = np.where(y > 0)[:trim]
        negidx = np.where(y == 0)[:trim]
        idx = np.append(posidx, negidx)
        y = y[idx]
        pred = pred[idx]
    y = y.astype(int)
    p = y[pred > 0]
    n = y[pred < 0]
    tp = np.sum(p)
    fn = np.sum(1 - p)
    tn = np.sum(1 - n)
    fp = np.sum(n)
    sense = tp / (fn + tp + 1e-4)
    speci = tn / (tn + fp + 1e-4)
    auc = metrics.roc_auc_score(y, pred)
    if np.sum(predint) * np.sum(1 - predint) != 0:
        f1 = metrics.f1_score(y, predint)
        mcc = metrics.matthews_corrcoef(y, predint)
    else:
        f1 = mcc = 0
    pr = metrics.average_precision_score(y, pred)
    result = np.array([auc, f1, mcc, pr, sense, speci])
    # result.append(auc)
    print('%s:\tauc: %.3f\tf1: %.3f\tmcc: %.3f\tap: %.3f\tsenseticity: %.3f\tspecificity: %.3f\t' % (
    prefix, auc, f1, mcc, pr, sense, speci), flush=True)
    return result

class Residual(nn.Module):
    ##### Residual on raw weighted sum, performs BatchNorm,Activation,model #####
    def __init__(self,model,append_dim=1,bypass=nn.Identity()):
        super(Residual, self).__init__()
        self.model=model
        self.bypass=bypass
        self.append_dim=append_dim
    def forward(self,x):
        out=self.model(x)
        bypass=self.bypass(x)
        if out.size()==bypass.size():
            out+=bypass
        else:
            out=torch.cat([out,bypass],dim=self.append_dim)
        return out
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

        self.fakemodel=nn.Identity()
    def forward(self, x):
        return x.view(*self.shape)
def PE(x,sig_denom=100,dim=-2):
    xsize=list(x.size())
    idxlength=np.ones(len(xsize))
    idxlength[dim]=-1
    seqlen=x.size()[dim]
    idx=torch.arange(seqlen).float()
    idx=torch.reshape(idx,list(idxlength))
    idx_perc=idx/seqlen
    idx_abs=(idx-idx//2)/sig_denom
    out=torch.cat([idx_perc,idx_abs],dim=-1)
    out=out=out.to(x.device)
    xsize=xsize[:-1]
    xsize.append(-1)
    out=out.expand(xsize)
    out=torch.cat([x,out],dim=-1)
    return out
class DilationResidualConv2d(nn.Module):
    def __init__(self,in_chan,out_chan,kernel_size=3,stride=1,dilation=1,convdrop=0.1,padding=-1):
            super(DilationResidualConv2d, self).__init__()
            if type(kernel_size)!=list:
                kernel_size=[kernel_size,kernel_size]
            if type(dilation)!=list:
                dilation=[dilation,dilation]
            if padding==-1:
                padding=[(kernel_size[0]-1)*(dilation[0]-1)//2,(kernel_size[1]-1)*(dilation[1]-1)//2]
            self.model=Residual(nn.Sequential(
                nn.Conv2d(in_channels=in_chan,out_channels=out_chan,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(out_chan),
                nn.LeakyReLU(),
                nn.Dropout2d(p=convdrop),
            ),
            bypass=nn.Sequential(
                nn.Conv2d(in_channels=in_chan,out_channels=out_chan,dilation=dilation,
                                   padding=padding,kernel_size=kernel_size,stride=stride),
                nn.BatchNorm2d(out_chan),
                nn.LeakyReLU(),
                nn.Dropout2d(p=convdrop),
            ))
    def forward(self,x):
        return self.model(x)
class GQA(nn.Module):
##### Batch first #####
##### [...,S,D] where S means Sequence and D means dimension #####
    def __init__(self,inputsize=512,decoder_inputsize=None,groupnum=8,headspergroup=16,permute=None,mpermute=None,
                 selfatt=False,dim=None,qkdim=128,vdim=128,ffoutsize=None,ffdropout=0.25,attdropout=0.5,addpositiondim=False):
        super(GQA, self).__init__()
        if decoder_inputsize==None:
            decoder_inputsize=inputsize
        if ffoutsize is None:
            ffoutsize=inputsize
        if attdropout!=0:
            self.attdropout=nn.Dropout(attdropout)
        else:
            self.attdropout=nn.Identity()

        if dim is not None:
            qkdim,vdim=dim,dim
        self.selfatt=selfatt
        self.permute=permute
        self.mpermute=mpermute
        self.pe=False
        if addpositiondim:
            self.pe=True
            inputsize+=2
            decoder_inputsize+=2
        if selfatt:
            self.Wqkv = nn.Linear(inputsize, (headspergroup+1)*groupnum* qkdim+groupnum* vdim)
        else:
            self.Wq = nn.Linear(inputsize, headspergroup*groupnum* qkdim)
            self.Wkv = nn.Linear(decoder_inputsize, groupnum* (vdim+qkdim))
        self.qkdim=qkdim
        self.vdim=vdim
        self.headspergroup=headspergroup
        self.groupnum=groupnum
        self.size = 1 / (qkdim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.Wz=nn.Sequential(nn.Linear(vdim*headspergroup*groupnum,ffoutsize),nn.LeakyReLU(),nn.Dropout(ffdropout))
    def forward(self,x,m=None,return_attention=False):
        if self.permute is not None:
            x=x.permute(self.permute)
        assert len(x.size())>2
        if self.pe:
            x=PE(x)
        xsize=list(x.size())
        if m is None:
            m = x
        else:
            assert len(m.size()) >2
            if self.mpermute is not None:
                m=m.permute(self.mpermute)
                m=PE(m)
        msize=list(m.size())
        qsize=xsize[:-1]+[self.groupnum, self.headspergroup, self.qkdim]
        ksize=msize[:-1]+[self.groupnum, self.qkdim]
        vsize=msize[:-1]+[self.groupnum, self.vdim]
        if self.selfatt:
            qkv=self.Wqkv(x)
            q = qkv[...,:self.headspergroup*self.groupnum*self.qkdim].view(qsize)
            k = qkv[...,self.headspergroup*self.groupnum*self.qkdim:self.headspergroup*self.groupnum*self.qkdim+self.groupnum*self.qkdim].view(ksize)
            v = qkv[...,self.headspergroup*self.groupnum*self.qkdim+self.groupnum*self.qkdim:].view(vsize)
        else:
            kv=self.Wkv(m)
            q = self.Wq(x).view(qsize)
            k = kv[...,:self.groupnum*self.qkdim].view(ksize)
            v = kv[...,self.groupnum*self.qkdim:].view(vsize)
        qk=torch.einsum('...sghd,...Sgd->...ghsS',q,k)
        weight = self.softmax(self.attdropout(torch.mul(qk, self.size)))
        if return_attention:
            return weight
        else:
            z=torch.einsum('...ghsS,...Sgd->...sghd',weight,v)
            z=z.flatten(-3)
            out=self.Wz(z)
            return out


class GeneVAE(pl.LightningModule):
    def __init__(self, input_size=28278, lambda_diag=10.,
                 lambda_offdiag=5.,
                 lr=2e-5, ):
        super(GeneVAE, self).__init__()
        self.learning_rate = lr
        self.acc = C.BinaryAccuracy()
        self.ap = C.BinaryAveragePrecision()
        self.mcc = C.BinaryMatthewsCorrCoef()
        self.auc = C.BinaryAUROC()
        self.spec = C.BinarySpecificity()
        self.sens = C.BinaryPrecision()
        self.f1 = C.BinaryF1Score()
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        super(GeneVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 1000), nn.LeakyReLU(), nn.Linear(1000, 500), nn.LeakyReLU())
        self.decoder = nn.Sequential(nn.Linear(500, 1000), nn.LeakyReLU(), nn.Linear(1000, input_size), nn.LeakyReLU())
        self.fc_mu = nn.Linear(500, 500)
        self.fc_var = nn.Linear(500, 500)

        def encode(self, input):
            """
            Encodes the input by passing through the encoder network
            and returns the latent codes.
            :param input: (Tensor) Input tensor to encoder [N x C x H x W]
            :return: (Tensor) List of latent codes
            """
            input=self.encoder(input)
            # Split the result into mu and var components
            # of the latent Gaussian distribution
            mu = self.fc_mu(input)
            log_var = self.fc_var(input)
            return [mu, log_var]
        def decode(self, input):
            """
            Maps the given latent codes
            onto the image space.
            :param z: (Tensor) [B x D]
            :return: (Tensor) [B x C x H x W]
            """
            input=self.decoder(input)
            return input

        def reparameterize(self, mu, logvar):
            """
            Reparameterization trick to sample from N(mu, var) from
            N(0,1).
            :param mu: (Tensor) Mean of the latent Gaussian [B x D]
            :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
            :return: (Tensor) [B x D]
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, input, **kwargs):
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
            return [self.decode(z), mu, log_var]

        def loss_function(self,
                          *args,
                          **kwargs) -> dict:
            """
            Computes the VAE loss function.
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
            :param args:
            :param kwargs:
            :return:
            """
            recons = args[0]
            reconsize = recons.size()
            input = args[1][..., :reconsize[-2], :reconsize[-1]]
            mu = args[2].flatten(1)
            log_var = args[3].flatten(1)

            #         kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
            recons_loss = F.mse_loss(recons, input, reduction='sum')
            if self.variation:

                kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

                # DIP Loss
                centered_mu = mu - mu.mean(dim=1, keepdim=True)  # [B x D]
                cov_mu = centered_mu.t().matmul(centered_mu).squeeze()  # [D X D]

                # Add Variance for DIP Loss II
                cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1=0), dim=0)  # [D x D]
                # For DIp Loss I
                # cov_z = cov_mu

                cov_diag = torch.diag(cov_z)  # [D]
                cov_offdiag = cov_z - torch.diag(cov_diag)  # [D x D]
                dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                           self.lambda_diag * torch.sum((cov_diag - 1) ** 2)
            else:
                kld_loss = 0
                dip_loss = 0
            loss = recons_loss * kld_loss + dip_loss
            # + kld_weight
            return {'loss': loss,
                    'Reconstruction_Loss': recons_loss,
                    'KLD': -kld_loss,
                    'DIP_Loss': dip_loss}

        def sample(self,
                   num_samples: int,
                   current_device: int, **kwargs):
            """
            Samples from the latent space and return the corresponding
            image space map.
            :param num_samples: (Int) Number of samples
            :param current_device: (Int) Device to run the model
            :return: (Tensor)
            """
            z = torch.randn(num_samples,
                            self.latent_dim)

            z = z.to(current_device)

            samples = self.decode(z)
            return samples

        def generate(self, x, **kwargs):
            """
            Given an input image x, returns the reconstructed image
            :param x: (Tensor) [B x C x H x W]
            :return: (Tensor) [B x C x H x W]
            """

            return self.forward(x)[0]

        def training_step(self, input, idx):
            [recon, mu, var] = self.forward(input)
            loss = self.loss_function(recon, input, mu, var)
            self.log("train_loss", loss['loss'])
            self.log("train_Recon_loss", loss['Reconstruction_Loss'])
            self.log("train_KLDivergence", loss['KLD'])
            self.log("train_DIP_loss", loss['DIP_Loss'])
            #         self.log('train_Accuracy', self.acc(out,y))
            #         self.log('train_AUROC', self.auc(out,y))
            return loss

        def validation_step(self, batch, idx):
            input = batch
            [recon, mu, var] = self.forward(input)
            loss = self.loss_function(recon, input, mu, var)
            #         self.log('validation_Accuracy', self.acc(out,y))
            #         self.log('validation_AUROC', self.auc(out,y))
            #         self.log('validation_Average_Precision', self.ap(out,y))
            #         self.log('validation_F1_score ', self.f1(out,y))
            #         self.log('validation_MCC', self.mcc(out,y))
            #         self.log('validation_Sensetivity', self.sens(out,y))
            #         self.log('validation_Specificity', self.spec(out,y
            self.log("valid_loss", loss['loss'])
            self.log("valid_Recon_loss", loss['Reconstruction_Loss'])
            self.log("valid_KLDivergence", loss['KLD'])
            self.log("valid_DIP_loss", loss['DIP_Loss'])
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

        def on_training_epoch_end(self, outs):
            # log epoch metric
            self.log("train_epoch_acc", self.acc)
class Show(nn.Module):
    def __init__(self):
        super(Show, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiAdaptPooling(nn.Module):
    def __init__(self, model, outsizelist=np.array([9, 25, 64])):
        super(MultiAdaptPooling, self).__init__()
        self.model = model
        self.modellist = []
        for i in outsizelist:
            self.modellist.append(nn.AdaptiveAvgPool1d(i))

    def forward(self, x):
        outlist = []
        for model in self.modellist:
            outlist.append(self.model(model(x)))
        out=torch.cat(outlist, -1)
        return out
class MultiModelAdaptPooling(nn.Module):
    def __init__(self, model, outsizelist=np.array([9, 25, 64])):
        super(MultiModelAdaptPooling, self).__init__()
        self.modellist = []
        for i in outsizelist:
            self.modellist.append(
                nn.Sequential(nn.AdaptiveAvgPool1d(i),
                              copy.deepcopy(model)).cuda())

    def forward(self, x):
        outlist = []
        for model in self.modellist:
            outlist.append(model(x))
        out = torch.cat(outlist, -1)
        return out
