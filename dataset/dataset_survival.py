from __future__ import print_function, division
import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self, csv_path='dataset_csv/ccrcc_clean.csv', mode='omic', apply_sig=False, 
                 seed=7, print_info=True, n_bins=4, ignore=[],
                 patient_strat=False, label_col=None, filter_dict={}, eps=1e-6, data_load_type='whole'):
        
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.data_load_type = data_load_type

        slide_data = pd.read_csv(csv_path, low_memory=False)

        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        
        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        # 删除重复行，获取有效数据
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        # 根据生存时间将数据分成指定的区间
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        # 构建患者到切片的映射字典
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array([slide_ids])
            else:
                slide_ids = slide_ids.values
            patient_dict[patient] = slide_ids
        
        # 移除已知的损坏切片
        slides_to_remove = {
            "TCGA-A7-A0CD": ["TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs"],
            "TCGA-A7-A6VX": ["TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs"]
        }

        for patient, bad_slides in slides_to_remove.items():
            if patient in patient_dict:
                patient_dict[patient] = np.array(
                    [s for s in patient_dict[patient] if s not in bad_slides], dtype=object
                )

        self.patient_dict = patient_dict
        
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        # 创建标签字典
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1
                
        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            slide_data.at[i, 'label'] = label_dict[(key, int(censorship))]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()

        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('/opt/data/private/MOTCat/datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [np.where(self.patient_data['label'] == i)[0] for i in range(self.num_classes)]
        self.slide_cls_ids = [np.where(self.slide_data['label'] == i)[0] for i in range(self.num_classes)]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            patient_labels.append(self.slide_data['label'][locations[0]])
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print(f"label column: {self.label_col}")
        print(f"label dictionary: {self.label_dict}")
        print(f"number of classes: {self.num_classes}")
        print("slide-level counts: \n", self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print(f'Patient-LVL; Number of samples registered in class {i}: {self.patient_cls_ids[i].shape[0]}')
            print(f'Slide-LVL; Number of samples registered in class {i}: {self.slide_cls_ids[i].shape[0]}')

    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(
                df_slice, metadata=self.metadata, mode=self.mode, 
                signatures=self.signatures, data_dir=self.data_dir, 
                label_col=self.label_col, patient_dict=self.patient_dict, 
                num_classes=self.num_classes, data_load_type=self.data_load_type
            )
        else:
            split = None
        return split

    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            
        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]
        
    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def _load_wsi_features(self, slide_ids, data_dir, replacement_suffix=None):
        """辅助函数：根据后缀加载切片特征，减少冗余代码"""
        if replacement_suffix:
            match = re.search(r'/data_feature/([A-Za-z0-9]+)_10x_448', data_dir)
            if match:
                cancer_type = match.group(1)
                data_dir = data_dir.replace(f"{cancer_type}_10x_448", f"{cancer_type}_{replacement_suffix}")
                
        features = []
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, 'pt_files', f"{slide_id.rstrip('.svs')}.pt")
            wsi_bag = torch.load(wsi_path, map_location=torch.device('cpu'))
            features.append(wsi_bag)
        
        return torch.cat(features, dim=0) if features else torch.empty(0)

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        data_dir = self.data_dir[self.slide_data['oncotree_code'][idx]] if isinstance(self.data_dir, dict) else self.data_dir
        
        if not self.use_h5 and data_dir:
            
            if self.mode == 'path':
                path_features = self._load_wsi_features(slide_ids, data_dir)
                return (path_features, torch.zeros((1,1)), label, event_time, c)

            elif self.mode == 'cluster':
                path_features = self._load_wsi_features(slide_ids, data_dir)
                cluster_ids = []
                for slide_id in slide_ids:
                    cluster_ids.extend(self.fname2ids[slide_id[:-4] + '.pt'])
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (path_features, torch.Tensor(cluster_ids), genomic_features, label, event_time, c)

            elif self.mode == 'omic':
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (torch.zeros((1,1)), genomic_features, label, event_time, c)

            elif self.mode == 'pathomic':
                path_features = self._load_wsi_features(slide_ids, data_dir)
                genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                return (path_features, genomic_features, label, event_time, c)

            elif self.mode == 'coattn':
                path_features_5x = torch.empty(0)
                path_features_10x = torch.empty(0)
                path_features_20x = torch.empty(0)
                
                if self.data_load_type == 'whole':
                    path_features_5x = self._load_wsi_features(slide_ids, data_dir)
                    path_features_10x = self._load_wsi_features(slide_ids, data_dir, replacement_suffix="10x_224")
                    path_features_20x = self._load_wsi_features(slide_ids, data_dir, replacement_suffix="40x_448")
                    
                elif self.data_load_type == 'part':
                    path_features_20x = self._load_wsi_features(slide_ids, data_dir, replacement_suffix="40x_448")
                    
                else:
                    raise NotImplementedError(f"Data load type '{self.data_load_type}' not implemented.")

                # 获取前6个 omic 特征并展开
                omics = [torch.tensor(self.genomic_features[self.omic_names[i]].iloc[idx].values) for i in range(6)]
                
                return (path_features_5x, path_features_10x, path_features_20x, 
                        *omics, label, event_time, c)
            else:
                raise NotImplementedError(f"Mode '{self.mode}' not implemented.")
        else:
            return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2, data_load_type=None):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.data_load_type = data_load_type

        self.slide_cls_ids = [np.where(self.slide_data['label'] == i)[0] for i in range(self.num_classes)]
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic + mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(list(set(omic) & set(self.genomic_features.columns)))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]

    def __len__(self):
        return len(self.slide_data)

    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)

    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed