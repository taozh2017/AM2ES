from __future__ import print_function
import argparse
import os
import torch
import numpy as np
from timeit import default_timer as timer

# Internal Imports
from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code


def get_parse_args():
    parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

    ### Checkpoint + Misc. Pathing Parameters
    parser.add_argument('--seed',            type=int, default=1,   help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k',               type=int, default=5,   help='Number of folds (default: 5)')
    parser.add_argument('--k_start',         type=int, default=-1,  help='Start fold (Default: -1, last fold)')
    parser.add_argument('--k_end',           type=int, default=-1,  help='End fold (Default: -1, first fold)')

    parser.add_argument('--data_root_dir',   type=str, default='/data_feature/GBMLGG_10x_448',  help='Data directory to WSI features')
    parser.add_argument('--results_dir',     type=str, default='./results',  help='Results directory')
    parser.add_argument('--which_splits',    type=str, default='5foldcv',  help='Which splits folder to use in ./splits/')
    parser.add_argument('--split_dir',       type=str, default='tcga_gbmlgg',  help='Which cancer type within ./splits/<which_splits> to use for training')
    parser.add_argument('--log_data',        action='store_true',   help='Log data using tensorboard')
    parser.add_argument('--overwrite',       action='store_true', default=True, help='Whether or not to overwrite experiments')

    ### Load Model Parameters
    parser.add_argument('--load_model',      action='store_true', default=False, help='whether to load model')
    parser.add_argument('--path_load_model', type=str, default='-', help='path of ckpt for loading')
    parser.add_argument('--start_epoch',     type=int, default=0, help='start_epoch.')
    parser.add_argument('--label',           default=False)
    parser.add_argument('--instance_eval',   default=False)

    ### Model Parameters.
    parser.add_argument('--model_type',      type=str, choices=['snn', 'amil', 'mcat', 'motcat', 'anchor'], default='anchor', help='Type of model')
    parser.add_argument('--data_load_type',  type=str, choices=['whole', 'part'], default='whole', help='slide-level classification loss function')
    parser.add_argument('--fusion',          type=str, choices=['None', 'concat'], default='concat', help='Type of fusion')
    parser.add_argument('--apply_sig',       action='store_true', default=True, help='Use genomic features as signature embeddings.')
    parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
    parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
    parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
    parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

    ### Optimizer Parameters + Survival Loss Function
    parser.add_argument('--opt',             type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--batch_size',      type=int, default=1,  help='Batch Size (Default: 1, due to varying bag sizes)')
    parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
    parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train')
    parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate')

    parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv','cox_surv', 'anchor_loss'], default='nll_surv', help='slide-level classification loss function')
    parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use')
    parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
    parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels')
    parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss')
    parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay')
    parser.add_argument('--alpha_surv',      type=float, default=0, help='How much to weigh uncensored patients')
    parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization')
    parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength')
    parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

    ### Micro-batch Parameters
    parser.add_argument('--use_micro_batch', action='store_true', default=False, help='Enable micro-batching')
    parser.add_argument('--bs_micro',        type=int, default=32000, help='The Size of Micro-batch')
    parser.add_argument('--explainer',       action='store_true', default=False, help='Enable explainer')

    return parser.parse_args()


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args, dataset):
    os.makedirs(args.results_dir, exist_ok=True)
    
    start_fold = 0 if args.k_start == -1 else args.k_start
    end_fold = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start_fold, end_fold)
    
    summary_all_folds = {}
    
    # Start 5-Fold CV Evaluation.
    for i in folds:
        start_t = timer()
        seed_torch(args.seed)
        
        args.results_pkl_path = os.path.join(args.results_dir, f'split_latest_val_{i}_results.pkl')
        
        if os.path.isfile(args.results_pkl_path) and not args.load_model and not args.overwrite:
            print(f"Skipping Split {i}")
            continue

        # Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(
            from_id=False, csv_path=f'{args.split_dir}/splits_{i}.csv'
        )
        print(f'Training: {len(train_dataset)}, Validation: {len(val_dataset)}')
        datasets = (train_dataset, val_dataset)

        ### Specify the input dimension size if using genomic features.
        if 'omic' in args.mode or args.mode in ['cluster', 'graph', 'pyramid']:
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print("Genomic Dimension", args.omic_input_dim)
        elif 'coattn' in args.mode:
            args.omic_sizes = train_dataset.omic_sizes
            print('Genomic Dimensions', args.omic_sizes)
        else:
            args.omic_input_dim = 0

        # Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            summary_results, print_results = train(datasets, i, args)
            
        # Write Results for Each Split to PKL
        save_pkl(args.results_pkl_path, summary_results)
        summary_all_folds[i] = print_results
        end_t = timer()
        print(f'Fold {i} Time: {end_t - start_t:.3f} seconds')

    print('=============================== summary ===============================')
    result_cindex = []
    for k, v in summary_all_folds.items():
        c_index = v['result'][0]
        print(f"Fold {k}, C-Index: {c_index:.4f}")
        result_cindex.append(c_index)
        
    result_cindex = np.array(result_cindex)
    print("Avg C-Index of {} folds: {:.3f}, stdp: {:.3f}, stds: {:.3f}".format(
        len(summary_all_folds), result_cindex.mean(), result_cindex.std(), result_cindex.std(ddof=1)))


if __name__ == "__main__":
    start_time = timer()
    
    args = get_parse_args()
    args = get_custom_exp_code(args)
    args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'

    seed_torch(args.seed)
    
    settings = {
        'data_root_dir': args.data_root_dir, 
        'num_splits': args.k,
        'k_start': args.k_start,
        'k_end': args.k_end,
        'task': args.task,
        'max_epochs': args.max_epochs,
        'results_dir': args.results_dir,
        'lr': args.lr,
        'experiment': args.exp_code,
        'reg': args.reg,
        'label_frac': args.label_frac,
        'bag_loss': args.bag_loss,
        'bag_weight': args.bag_weight,
        'seed': args.seed,
        'model_type': args.model_type,
        'model_size_wsi': args.model_size_wsi,
        'model_size_omic': args.model_size_omic,
        "use_drop_out": args.drop_out,
        'weighted_sample': args.weighted_sample,
        'gc': args.gc,
        'opt': args.opt,
        'bs_micro': args.bs_micro,
        'data_load_type': args.data_load_type
    }

    print('\nLoad Dataset')
    if 'survival' in args.task:
        args.n_classes = 4
        combined_study = '_'.join(args.task.split('_')[:2])
        data_path = os.path.dirname(args.data_root_dir)
        path = os.path.dirname(data_path)
        csv_path = f'{path}/{args.dataset_path}/{combined_study}_all_clean.csv.zip'
        
        dataset = Generic_MIL_Survival_Dataset(
            csv_path=csv_path,
            mode=args.mode,
            apply_sig=args.apply_sig,
            data_dir=args.data_root_dir,
            seed=args.seed,
            print_info=True,
            patient_strat=False,
            n_bins=4,
            label_col='survival_months',
            ignore=[],
            data_load_type=args.data_load_type
        )
    else:
        raise NotImplementedError

    if isinstance(dataset, Generic_MIL_Survival_Dataset):
        args.task_type = 'survival'
    else:
        raise NotImplementedError

    if args.model_type in ['motcat', 'anchor']:
        exp_code = f"{args.exp_code}_s{args.seed}"
    else:
        exp_code = args.exp_code

    print("===="*20)
    print("Experiment Name:", exp_code)
    print("===="*20)

    # Configure Directories
    args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
    os.makedirs(args.results_dir, exist_ok=True)
    print("logs saved at ", args.results_dir)

    data_path = os.path.dirname(os.path.dirname(args.data_root_dir))
    args.split_dir = os.path.join(data_path, 'splits', args.which_splits, args.split_dir)
    print("split_dir", args.split_dir)
    assert os.path.isdir(args.split_dir)
    
    settings.update({'split_dir': args.split_dir})

    with open(f'{args.results_dir}/experiment_{args.exp_code}.txt', 'w') as f:
        print(settings, file=f)

    print("################# Settings ###################")
    for key, val in settings.items():
        print(f"{key}:  {val}")

    # Run main logic
    main(args, dataset)
    
    end_time = timer()
    print("Finished!")
    print(f'Total Script Time: {end_time - start_time:.3f} seconds')