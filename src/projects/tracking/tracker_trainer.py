import torch
import torch.nn.functional as F
import wandb
import re

from collections import namedtuple
from functools import lru_cache
from ...handlers.trainer import Trainer
from ...loss.cross_entropy import CrossEntropyLoss

class TrackerTrainer(Trainer):
    #== Methods for tracking certain points =======================================================#
    @lru_cache(maxsize=5)
    def make_track_batch(self, bsz):
        race_M_dev = self.data_handler.prep_split('race-M', 'dev', lim=5)
        race_H_dev = self.data_handler.prep_split('race-H', 'dev', lim=5)
        race_C_dev = self.data_handler.prep_split('race-C', 'dev', lim=5)

        track_data = race_M_dev + race_H_dev + race_C_dev
        train_batches = self.batcher(
            data = track_data, 
            bsz = bsz, 
            shuffle = False
        )
        return list(train_batches)

    def track_load(self, bsz:int=4):
        # set model back to eval
        self.model.eval()

        # get probability of the label for every item in the tracker
        label_confidence = []

        track_batches = self.make_track_batch(bsz)
        for batch in track_batches:
            output = self.model_loss(batch)

            probs = F.softmax(output.logits, dim=-1)
            lab_prob = probs.gather(-1, batch.labels.unsqueeze(-1)).squeeze(-1)
            label_confidence += lab_prob.cpu().tolist()

        # log in wandb the probability
        confidences = {}
        for k, prob in enumerate(label_confidence):
            confidences[f'track-{k}'] = prob
        
        wandb.log(confidences)

        # set model back to train
        self.model.train()

    #== Original training method ==================================================================#
    def train(self, args: namedtuple):
        self.save_args('train-args.json', args)
 
        # set up optimization objects
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr)
        optimizer.zero_grad()

        # set up model
        self.to(args.device)
        self.model.train()
        self.log_num_params()
        self.model_loss = CrossEntropyLoss(self.model)
        
        # Reset loss metrics
        self.best_dev = (0, {})
        self.model_loss.reset_metrics()

        # Get train, val, test split of data
        train, dev, test = self.data_handler.prep_data(args.dataset, args.lim)

        # Setup wandb for online tracking of experiments
        if args.wandb: self.setup_wandb(args)

        for epoch in range(1, args.epochs+1):
            #== Training =============================================
            train_batches = self.batcher(
                data = train, 
                bsz = args.bsz, 
                shuffle = True
            )
            for step, batch in enumerate(train_batches, start = 1):
                #== Added Line for Tracking =======================================================#
                self.track_load()
                #==================================================================================#
                output = self.model_loss(batch)

                # update graidents, clip gradients, and update parameters
                optimizer.zero_grad()
                output.loss.backward()
                if args.grad_clip: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                optimizer.step()
                
                # Print train performance every log_every samples
                if step % (args.log_every//args.bsz) == 0:
                    metrics = self.get_metrics()
                    
                    self.log_metrics(
                        metrics = metrics,
                        mode = 'train', 
                        epoch = epoch,
                        ex_step = step*args.bsz
                    )

                    if args.wandb: self.log_wandb(metrics, mode='train')
                    self.model_loss.reset_metrics()   
            
            #== Validation ============================================
            metrics = self.validate(dev, mode = 'dev')
            self.log_metrics(metrics = metrics, mode = 'dev')
            if args.wandb: self.log_wandb(metrics, mode = 'dev')

            if metrics['acc'] > self.best_dev[1].get('acc', 0):
                self.best_dev = (epoch, metrics.copy())
                self.save_model()
            
            self.log_metrics(metrics=self.best_dev[1], mode='dev-best', epoch=self.best_dev[0])

            if epoch - self.best_dev[0] >= 3:
                break

    def setup_wandb(self, args: namedtuple):
        # remove everything before */trained_models for exp_name
        exp_name = re.sub(r'^.*?trained_models', '', self.exp_path)

        # remove the final -seed-i from the group name
        group_name = '/seed'.join(exp_name.split('/seed')[:-1])

        #init wandb project
        wandb.init(
            project=f"tracker-{args.dataset}",
            entity='mg-speech-group',
            group=group_name,
            name=exp_name, 
            dir=self.exp_path,
        )

        # save experiment config details
        cfg = {
            'dataset': args.dataset,
            'bsz': args.bsz,
            'lr': args.lr,
            'transformer': self.model_args.transformer,
            'formatting': self.model_args.formatting
        }

        wandb.config.update(cfg) 
        wandb.watch(self.model)
