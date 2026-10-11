import os
import numpy as np
import torch
import pandas as pd
from sksurv.metrics import concordance_index_censored

torch.set_num_threads(2)

def train_loop_survival_coattn_mb(epoch, bs_micro, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=32, args=None):
    model.train()
    train_loss_surv, train_loss = 0., 0.
    
    beta_sparse = 0.1
    lambda_anc = 0.05

    print('\n')
    all_risk_scores = np.zeros(len(loader))
    all_censorships = np.zeros(len(loader))
    all_event_times = np.zeros(len(loader))
    
    for batch_idx, batch_data in enumerate(loader):
        # Unpack batch and move to GPU
        data_5x, data_10x, data_20x = batch_data[0].cuda(), batch_data[1].cuda(), batch_data[2].cuda()
        data_omic1 = batch_data[3].type(torch.FloatTensor).cuda()
        data_omic2 = batch_data[4].type(torch.FloatTensor).cuda()
        data_omic3 = batch_data[5].type(torch.FloatTensor).cuda()
        data_omic4 = batch_data[6].type(torch.FloatTensor).cuda()
        data_omic5 = batch_data[7].type(torch.FloatTensor).cuda()
        data_omic6 = batch_data[8].type(torch.FloatTensor).cuda()
        label = batch_data[9].type(torch.LongTensor).cuda()
        event_time = batch_data[10]
        c = batch_data[11].type(torch.FloatTensor).cuda()

        hazards, S, Y_hat, A = model(
            x_path_5x=data_5x, x_path_10x=data_10x, x_path_20x=data_20x, 
            x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, 
            x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6, 
            label=label, c=c
        )

        # Survival loss computation
        if args.bag_loss == 'nll_surv':
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        elif args.bag_loss == 'cox_surv':
            loss = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
        else:
            raise NotImplementedError
            
        # Anchor regularization
        orth_5x, sparse_5x = A['orth_loss_5x'], A['sparse_loss_5x']
        orth_10x, sparse_10x = A['orth_loss_10x'], A['sparse_loss_10x']
        orth_20x, sparse_20x = A['orth_loss_20x'], A['sparse_loss_20x']
        
        anc_reg = (orth_5x + orth_10x + orth_20x) + beta_sparse * (sparse_5x + sparse_10x + sparse_20x)
        loss = loss + lambda_anc * anc_reg
        
        loss_reg = reg_fn(model) * lambda_reg if reg_fn else 0.

        # Record metrics
        risk = -torch.sum(S, dim=1).item()
        loss_value = loss.item()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time
        
        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 50 == 0:
            train_batch_str = f'batch {batch_idx}, loss: {loss_value:.4f}, label: {label.item()}, event_time: {float(event_time):.4f}, risk: {float(risk):.4f}'
            print(train_batch_str)
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str + '\n')

        # Backward and Optimize
        loss = (loss + loss_reg) / gc
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # Calculate loss and C-index for the epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    train_epoch_str = f'Epoch: {epoch}, train_loss_surv: {train_loss_surv:.4f}, train_loss: {train_loss:.4f}, train_c_index: {c_index_train:.4f}'
    print(train_epoch_str)
    
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str + '\n')

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)


def validate_survival_coattn_mb(cur, epoch, bs_micro, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    risk_scores_list = []
    censorships_list = []
    event_times_list = []
    
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    sample_counter = 0

    for batch_idx, batch_data in enumerate(loader):
        # Unpack batch and move to GPU
        data_5x, data_10x, data_20x = batch_data[0].cuda(), batch_data[1].cuda(), batch_data[2].cuda()
        data_omic1 = batch_data[3].type(torch.FloatTensor).cuda()
        data_omic2 = batch_data[4].type(torch.FloatTensor).cuda()
        data_omic3 = batch_data[5].type(torch.FloatTensor).cuda()
        data_omic4 = batch_data[6].type(torch.FloatTensor).cuda()
        data_omic5 = batch_data[7].type(torch.FloatTensor).cuda()
        data_omic6 = batch_data[8].type(torch.FloatTensor).cuda()
        label = batch_data[9].type(torch.LongTensor).cuda()
        event_time = batch_data[10]
        c = batch_data[11].type(torch.FloatTensor).cuda()
        
        with torch.no_grad():
            hazards, S, Y_hat, A = model(
                x_path_5x=data_5x, x_path_10x=data_10x, x_path_20x=data_20x, 
                x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, 
                x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6, 
                label=label, c=c
            )
                    
            # Survival loss computation
            if args.bag_loss == 'nll_surv':
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
            elif args.bag_loss == 'cox_surv':
                loss_micro = loss_fn(hazards=hazards.squeeze(), S=S, c=c)
            else:
                raise NotImplementedError
                
            loss_value = loss_micro.item()

            # Anchor regularization
            beta_sparse = 0.1
            lambda_anc = 0.5
            orth_5x, sparse_5x = A.get('orth_loss_5x', 0), A.get('sparse_loss_5x', 0)
            orth_10x, sparse_10x = A.get('orth_loss_10x', 0), A.get('sparse_loss_10x', 0)
            orth_20x, sparse_20x = A.get('orth_loss_20x', 0), A.get('sparse_loss_20x', 0)
            
            anc_reg = (orth_5x + orth_10x + orth_20x) + beta_sparse * (sparse_5x + sparse_10x + sparse_20x)
            anc_reg_value = anc_reg.item() if torch.is_tensor(anc_reg) else anc_reg

            risk_batch = -torch.sum(S, dim=1).cpu().numpy()

        loss_reg = reg_fn(model) * lambda_reg if reg_fn else 0.

        # Data collection for metrics calculation
        risk_scores_list.extend(risk_batch)
        censorships_list.extend(c.cpu().numpy())
        
        if torch.is_tensor(event_time):
            event_times_list.extend(event_time.cpu().numpy().reshape(-1))
        else:
            event_times_list.extend(np.atleast_1d(event_time).reshape(-1))

        current_batch_len = len(risk_batch)
        for i in range(current_batch_len):
            if sample_counter < len(slide_ids):
                s_id = slide_ids.iloc[sample_counter]
                patient_results[s_id] = {
                    'slide_id': np.array(s_id), 
                    'risk': risk_batch[i], 
                    'disc_label': label[i].item(), 
                    'survival': event_time[i].item() if torch.is_tensor(event_time) else event_time[i], 
                    'censorship': c[i].item()
                }
                sample_counter += 1 

        val_loss_surv += loss_value
        val_loss += loss_value + lambda_anc * anc_reg_value + loss_reg

    all_risk_scores = np.array(risk_scores_list)
    all_censorships = np.array(censorships_list)
    all_event_times = np.array(event_times_list)

    if len(all_risk_scores) == 0:
        print("\n[Warning] 验证集为空 (0 samples)！跳过 C-index 计算，默认返回 0.5。")
        return patient_results, 0.5, False

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    val_epoch_str = f"val c-index: {c_index:.4f}"
    
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str + '\n')
    
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        ckpt_name = os.path.join(results_dir, f"s_{cur}_minloss_checkpoint.pt")
        early_stopping(epoch, val_loss_surv, model, ckpt_name=ckpt_name)
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False