import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from vae import VAE, IWAE, LMCVAE
from utils import make_dataloaders, get_activations


def main(vae_model="LMCVAE"):
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/') 

    act_func = get_activations()
    kwargs = {'num_workers': 20, 'pin_memory': True}
    train_loader, val_loader = make_dataloaders(dataset='cifar',
                                                batch_size=32,
                                                val_batch_size=50,
                                                binarize=False,
                                                **kwargs)
    image_shape = train_loader.dataset.shape_size
    if vae_model == "VAE":
        model = VAE(shape=image_shape, act_func=act_func["gelu"],
                    num_samples=1, hidden_dim=64,
                    net_type="conv", dataset='cifar', specific_likelihood=None,
                    sigma=1.)
    elif vae_model == "IWAE":
        model = IWAE(shape=image_shape, act_func=act_func["gelu"], num_samples=1,
                     hidden_dim=64,
                     name=vae_model, net_type="conv", dataset='cifar',
                     specific_likelihood=None, sigma=1.)
    elif vae_model == 'LMCVAE':
        model = LMCVAE(shape=image_shape, step_size=0.01, K=3,
                       num_samples=1, acceptance_rate_target=0.95,
                       dataset='cifar', net_type="conv", act_func=act_func["gelu"],
                       hidden_dim=64, name=vae_model, grad_skip_val=0.,
                       grad_clip_val=0., use_score_matching=False,
                       use_cloned_decoder=False, learnable_transitions=False,
                       variance_sensitive_step=False,
                       ula_skip_threshold=0., annealing_scheme='linear',
                       specific_likelihood=None, sigma=1.)
    else:
        raise ValueError


    trainer = pl.Trainer(max_epochs=50, logger=tb_logger, fast_dev_run=False)
    pl.Trainer()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()