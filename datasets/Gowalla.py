import numpy as np
import pandas as pd
from typing import Callable, List, Optional

import os.path as osp
import os

import torch

from torch_geometric.data import HeteroData, InMemoryDataset, download_url

class Gowalla(InMemoryDataset):
    url = 'https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz'
  

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        core: int = None,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.core = core
        self.load(self.processed_paths[0], data_cls=HeteroData)
    @property
    def raw_file_names(self) -> List[str]:
        return ['loc-gowalla_totalCheckins.txt', 'filtered_total.txt', 'user_id_map.txt', 'item_id_map.txt', 'train.txt', 'test.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    @property
    def ziped_file_name(self) -> str:
        return 'loc-gowalla_totalCheckins.txt.gz'

    def download(self) -> None:
        import gzip

        if not osp.isfile(osp.join(self.root, 'loc-gowalla_totalCheckins.txt.gz')):
            download_url(f'{self.url}', self.root)
        if not osp.isfile(osp.join(self.root,'raw/loc-gowalla_totalCheckins.txt')): 
            os.system(f'gzip -d {osp.join(self.root,"loc-gowalla_totalCheckins.txt.gz")}')
            os.system(f'mv {osp.join(self.root,"raw/loc-gowalla_totalCheckins.txt")} {osp.join(self.root,"loc-gowalla_totalCheckins.txt.gz")}')
    
    def core_filter(self, df, threshold):

        df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]),sep = '\t', names = ['user', 'time', 'long', 'lat', 'item'])
        
        filtered_df = self.filtering(df, threshold)
        processed_df, user_id, item_id  = self.refactoring_from_0(filtered_df)

        processed_df.to_csv(osp.join(self.raw_dir, self.raw_file_names[1]),sep=" ", index=False, header=None)
        user_id.to_csv(osp.join(self.raw_dir, self.raw_file_names[2]),sep=" ", index=False, header=None)
        item_id.to_csv(osp.join(self.raw_dir, self.raw_file_names[3]),sep=" ", index=False, header=None)

        return processed_df

    def refactoring_from_0(self, df):
        out_df = pd.DataFrame() 
        
        original_uid = np.sort(df['user'].unique())
        original_iid = np.sort(df['item'].unique())

        u_range = range(len(original_uid))
        i_range = range(len(original_iid))

        uid_mapping = { o_id: n_id for o_id, n_id in zip(original_uid, u_range)} # 원래 유저 아이디 (중간중간 비어있음) : 순서대로 유저 아이디
        iid_mapping = { o_id: n_id for o_id, n_id in zip(original_iid,i_range)} # 원래 아이템 아이디 : 순서대로 아이템 아이디

        uid_map = pd.DataFrame({'o_id' : list(uid_mapping.keys()), 'n_id' : list(uid_mapping.values())})

        iid_map = pd.DataFrame({'o_id' : list(iid_mapping.keys()),'n_id':list(iid_mapping.values())})


        out_df['user'] = df['user'].map(uid_mapping)
        out_df['item'] = df['item'].map(iid_mapping)
        out_df['time'] = df['time']
        return out_df, uid_map, iid_map

    def filtering(self, df, threshold) :
        # Remove duplicate
        fdf = df.drop_duplicates(subset=['user', 'item'], keep='first')

        # Filter the date which have less interaction than threshold
        while fdf.user.value_counts().min() < threshold or fdf.item.value_counts().min() < threshold:
            # Item Filter
            df_item = fdf.groupby('item').count()
            df_item = df_item[df_item.user < threshold]
            li = df_item.index.to_list()
            fdf = fdf.drop(fdf.loc[fdf.item.isin(li)].index)

            # User Filter
            df_usr = fdf.groupby('user').count()
            df_usr = df_usr[df_usr.item < threshold]
            li = df_usr.index.to_list()
            fdf = fdf.drop(fdf.loc[fdf.user.isin(li)].index)
            
        # Check final result
        # print(f"Total Edges : {len(fdf)}\nTotal User : {len(fdf['user'].unique())}\nTotal item : {len(fdf['item'].unique())} \
        #             \nMin Interaction Per user : {fdf.user.value_counts().min()} \
        #             \nMax Interaction Per user : {fdf.user.value_counts().max()} \
        #             \nAvg Interaction Per user : {fdf.user.value_counts().mean()}\
        #             \nMin Interaction Per item : {fdf.item.value_counts().min()} \
        #             \nMax Interaction Per item : {fdf.item.value_counts().max()} \
        #             \nAvg Interaction Per item : {fdf.item.value_counts().mean()}")
    
        fdf = fdf.reset_index().drop(columns = ['index'])
        return fdf

    def process(self) -> None:
        from sklearn.model_selection import train_test_split

        data = HeteroData()
        attr_names = ['edge_index', 'edge_label_index']
        # Process number of nodes for each node type:
        node_types = ['user', 'item']

        # If there train.txt and test.txt is existed
        if osp.isfile(osp.join(self.root,f"raw/{self.raw_file_names[4]}")) \
            and osp.isfile(osp.join(self.root,f"raw/{self.raw_file_names[5]}")):
            # Process edge information for training and testing:
            # dataset to pyg data format
            for path, node_type in zip(self.raw_paths[2:4], node_types):
                df = pd.read_csv(path, sep=' ', header= None)
                data[node_type].num_nodes = len(df)

            for path, attr_name in zip(self.raw_paths[4:], attr_names):
                temp_df = pd.read_csv(path, names = ['user', 'item', 'time'], header = None)
                rows = temp_df['user'].values
                cols = temp_df['item'].values
                index = torch.tensor([rows, cols])

                data['user', 'rates', 'item'][attr_name] = index
                if attr_name == 'edge_index':
                    data['item', 'rated_by', 'user'][attr_name] = index.flip([0])
        # If there train.txt and test.txt is not existed
        else:
            # If already filterd file existed
            if osp.isfile(osp.join(self.root,f"raw/{self.raw_file_names[1]}")) \
                and osp.isfile(osp.join(self.root,f"raw/{self.raw_file_names[2]}")) \
                    and osp.isfile(osp.join(self.root,f"raw/{self.raw_file_names[3]}")):
                df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[1]), sep = " ", names = ['user', 'item', 'time'], header = None)

            # If already filterd file not existed
            else:
                df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]),  names = ['user', 'time', 'long', 'lat', 'item'], header = None)
                self.core_filter(df, self.core)
                df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[1]), sep = " ", names = ['user', 'item', 'time'], header = None)
            # train test split(randomly)
            """
            TODO ::
            시드를 사용자가 지정해서 tr test 나눌수 있게 수정
            """
            tr, test = train_test_split(df, test_size = 0.2)
            tr.to_csv(osp.join(self.raw_dir, self.raw_file_names[4]), index = False,header = False)
            test.to_csv(osp.join(self.raw_dir, self.raw_file_names[5]), index = False, header = False)

            # dataset to pyg data format
            for path, node_type in zip(self.raw_paths[2:4], node_types):
                df = pd.read_csv(path, sep=' ', header= None)
                data[node_type].num_nodes = len(df)

            for temp_df, attr_name in zip([tr,test],attr_names):
                rows = temp_df['user'].values
                cols = temp_df['item'].values
                index = torch.tensor([rows, cols])

                data['user', 'rates', 'item'][attr_name] = index
                if attr_name == 'edge_index':
                    data['item', 'rated_by', 'user'][attr_name] = index.flip([0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
