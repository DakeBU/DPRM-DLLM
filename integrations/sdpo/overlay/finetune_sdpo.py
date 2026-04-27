from utils import set_seed
from tqdm import tqdm
import lightning as L
from argparse import ArgumentParser
import os
from hydra import initialize, compose
from sdpo_gosai import DiffusionSDPO
from lightning import Trainer
from dataloader_gosai import GosaiDataset, get_dataloaders_gosai
import torch
from eval import eval_model
import wandb

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='data_and_model/')
    parser.add_argument('--model_path', type=str, default='mdlm/outputs_gosai/pretrained.ckpt')
    parser.add_argument('--ref_model_path', type=str, default='mdlm/outputs_gosai/pretrained.ckpt')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--K', type=int, default=2000)
    parser.add_argument('--config_path', type=str, default='configs_gosai')
    parser.add_argument('--config_name', type=str, default='config_sdpo_gosai.yaml')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--save_path', type=str, default='results.pt')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--generator', type=str, default='rkl')
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--skip_final_inline_eval', action='store_true')
    parser.add_argument('--order_policy', type=str, default='baseline',
                        choices=['baseline', 'progressive', 'dprm', 'dprm_random'])
    parser.add_argument('--dprm_beta', type=float, default=1.0)
    parser.add_argument('--dprm_warmup_steps', type=int, default=100)
    parser.add_argument('--dprm_switch_steps', type=int, default=400)
    parser.add_argument('--dprm_ready_count', type=int, default=64)
    parser.add_argument('--dprm_shortlist_size', type=int, default=64)
    args = parser.parse_args()
    
    set_seed(args.seed, use_cuda=True)
    
    initialize(config_path=args.config_path, job_name="load_model")
    config = compose(config_name=args.config_name)
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    config.ordering = {
        'policy': args.order_policy,
        'dprm_beta': args.dprm_beta,
        'dprm_warmup_steps': args.dprm_warmup_steps,
        'dprm_switch_steps': args.dprm_switch_steps,
        'dprm_ready_count': args.dprm_ready_count,
        'dprm_phase_bins': 8,
        'dprm_conf_bins': 10,
        'dprm_shortlist_size': args.dprm_shortlist_size,
    }
    
    if args.wandb:
        wandb.init(project=os.environ.get('WANDB_PROJECT', 'DNA-FPO'),
                   group=os.environ.get('WANDB_GROUP', 'fPO'),
                   name=os.environ.get('WANDB_NAME', None),
                   reinit=True, settings=wandb.Settings(start_method='fork'),
                   config=vars(args))
    
    model_path = os.path.join(args.base_path, args.model_path)
    ref_path = os.path.join(args.base_path, args.ref_model_path)
    
    ref_model = DiffusionSDPO.load_from_checkpoint(ref_path, config=config, beta=1.0, generator=args.generator, strict=False).to('cuda')
    ref_model.eval()
    
    model = DiffusionSDPO.load_from_checkpoint(model_path, config=config, beta=args.beta, strict=False)
    model.train()
    model._manual_global_step = 0
    
    model.set_ref_model(ref_model)
    
    sdpo_dataset = GosaiDataset()
    sdpo_train_loader = torch.utils.data.DataLoader(
        sdpo_dataset,
        batch_size=args.K,
        shuffle=True
    )
    
    optim = torch.optim.AdamW(
      model.parameters(),
      lr=args.lr,
      betas=(args.beta1,
             args.beta2),
      eps=args.eps,
      weight_decay=args.weight_decay)
    
    pbar = tqdm(range(args.num_epochs))
    device = torch.device('cuda')
    
    for i in pbar:
        total_loss = 0.
        pbar2 = tqdm(enumerate(sdpo_train_loader))
            
        for idx, batch in pbar2:
            batch['seqs'] = batch['seqs'].to(device)
            batch['clss'] = batch['clss'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            
            optim.zero_grad()
            
            loss = model.training_step(batch, idx)
            loss.backward()
            
            if config.trainer.gradient_clip_val > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.trainer.gradient_clip_val)
                
            optim.step()
            model._manual_global_step = getattr(model, '_manual_global_step', 0) + 1
            total_loss += loss.item()
            
            pbar2.set_description(f'Batch: {idx}. Train loss: {loss.item()}')
            
        avg_loss = total_loss / len(sdpo_train_loader)
        pbar.set_description(
            (
                f'Epoch: {i}. Avg. Train loss: {avg_loss}'
            )
        )
        
        if (i + 1) % args.eval_every == 0:
            model.eval()
            all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef = eval_model(model, ref_model, 10, 64, args.verbose)
            if wandb.run is not None:
                wandb.log({'pred_hepg2': generated_preds[:, 0].mean()})
                wandb.log({'log_lik': model_logl.mean()})
                wandb.log({'atac': generated_atac_acc})
                wandb.log({'p_coef': generated_p_coef})
            model.train()
        
    model.eval()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")

    if args.skip_final_inline_eval:
        return

    try:
        all_detokenized_samples, model_logl, generated_preds, generated_atac_acc, generated_p_coef = eval_model(model, ref_model, 10, 64, args.verbose)
        if wandb.run is not None:
            wandb.log({'pred_hepg2': generated_preds[:, 0].mean()})
            wandb.log({'log_lik': model_logl.mean()})
            wandb.log({'atac': generated_atac_acc})
            wandb.log({'p_coef': generated_p_coef})
            wandb.log({'reward_atac': generated_preds[:, 0].mean() * generated_atac_acc})
            wandb.log({'total_metric': generated_preds[:, 0].mean() * generated_atac_acc * generated_p_coef})
            wandb.log({'reward_pearson': generated_preds[:, 0].mean() * generated_p_coef})
    except Exception as exc:
        print(f"Final inline eval failed after saving model: {exc}")
