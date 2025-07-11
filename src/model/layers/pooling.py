import torch
import torch_scatter

from torch_cluster import grid_cluster
from torch_geometric.nn.pool.avg_pool import _avg_pool_x
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.data import Data, Batch
from .components import BatchNormData
from typing import List, Callable


def consecutive_cluster(src):
    """
    不連続な可能性のあるクラスタIDを、0から始まる連続した整数に再マッピングする関数。
    これにより、後続の処理（特にscatter操作）が効率的に行える。
    """
    # srcからユニークなクラスタIDとその逆引きインデックス、出現回数を取得
    unique, inv, counts = torch.unique(src, sorted=True, return_inverse=True, return_counts=True)
    # 元のノードを新しいクラスタID順にソートするためのインデックス(perm)を作成
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    return unique, inv, perm, counts


class Pooling(torch.nn.Module):
    """
    グラフデータを空間的なグリッドに基づいてプーリング（ダウンサンプリング）する層。
    CNNにおけるプーリング層と同様に、グラフを粗くして計算量を削減し、受容野を広げる役割を持つ。
    """
    def __init__(self, size: List[float], width, height, batch_size, transform: Callable[[Data, ], Data], aggr: str = 'max', keep_temporal_ordering=False, dim=2, self_loop=False, in_channels=-1):
        super(Pooling, self).__init__()
        assert aggr in ['mean', 'max'] # 集約方法は 'mean' または 'max' のみサポート
        self.aggr = aggr

        # プーリングの単位となるボクセル(グリッド)のサイズを登録。バッチ次元用に末尾に1を追加。
        self.register_buffer("voxel_size", torch.cat([size, torch.Tensor([1])]), persistent=False)

        # プーリング後に適用する変換処理（例：エッジ属性の計算など）
        self.transform = transform
        # 時間的な順序を維持するかどうかのフラグ
        self.keep_temporal_ordering = keep_temporal_ordering
        self.dim = dim # 空間次元（通常は2D）

        # grid_cluster関数に渡す、クラスタリングを行う空間の開始点と終了点
        self.register_buffer("start", torch.Tensor([0,0,0,0]), persistent=False)
        self.register_buffer("end", torch.Tensor([0.9999999,0.9999999,0.9999999,batch_size-1]), persistent=False)
        # 座標を正規化/非正規化するための逆数 (1/width, 1/height)
        self.register_buffer("wh_inv", 1/torch.Tensor([[width, height]]), persistent=False)
        
        # このプーリング層で生成されうるボクセルの最大数を計算
        self.max_num_voxels = batch_size * self.num_grid_cells
        self.register_buffer("sorted_cluster", torch.arange(self.max_num_voxels), persistent=False)

        # プーリング後のグラフで自己ループを許容するかどうか
        self.self_loop = self_loop

        # 入力チャネル数が指定されていれば、バッチ正規化レイヤーを初期化
        self.bn = None
        if in_channels > 0:
            self.bn = BatchNormData(in_channels)

    @property
    def num_grid_cells(self):
        """このプーリング層で生成される可能性のあるグリッドセルの総数を計算するプロパティ。"""
        # (1 / ボクセルサイズ) の積で総数を算出
        return (1/self.voxel_size+1e-3).int().prod()
    
    def round_to_pixel(self, pos, wh_inv):
        """プーリング後の座標を、それが属するボクセルのグリッド座標に丸める関数。"""
        # (pos / wh_inv) の床関数を計算し、再度 wh_inv を掛けることで丸める
        torch.div(pos+1e-5, wh_inv, out=pos, rounding_mode='floor')
        return pos * wh_inv

    def forward(self, data: Data):
        """プーリング処理の本体。"""
        # 入力グラフが空の場合は何もせずにそのまま返す
        if data.x.shape[0] == 0:
            return data

        # クラスタリングのため、ノードの座標(pos)にバッチ情報(batch)を結合する
        pos = torch.cat([data.pos, data.batch.float().view(-1,1)], dim=-1)
        
        # 座標を指定されたボクセルサイズでグリッドに分割し、各ノードが属するクラスタIDを計算
        cluster = grid_cluster(pos, size=self.voxel_size, start=self.start, end=self.end)
        
        # クラスタIDを0から始まる連続した整数に変換し、ソート用のインデックス(perm)も得る
        unique_clusters, cluster, perm, _ = consecutive_cluster(cluster)
        
        # 元のエッジ(data.edge_index)を、新しいクラスタIDベースのエッジに変換する
        edge_index = cluster[data.edge_index]
        
        if self.self_loop:
            # 自己ループを許容する場合、重複したエッジを削除する
            edge_index = edge_index.unique(dim=-1)
        else:
            # 自己ループ(始点と終点が同じクラスタのエッジ)を削除する
            edge_index = edge_index[:, edge_index[0]!=edge_index[1]]
            if edge_index.shape[1] > 0:
                # 重複したエッジを削除する
                edge_index = edge_index.unique(dim=-1)

        # 新しいノード（クラスタ）に対応するバッチIDを計算
        batch = None if data.batch is None else data.batch[perm]
        # 新しいノードの座標を、クラスタ内のノード座標の平均値として計算
        pos = None if data.pos is None else pool_pos(cluster, data.pos)

        if self.keep_temporal_ordering:
            # 各クラスタの代表時刻（＝クラスタ内の最大時刻）を計算
            t_max, _ = torch_scatter.scatter_max(data.pos[:,-1], cluster, dim=0)
            # 各エッジの始点と終点のクラスタの代表時刻を取得
            t_src, t_dst = t_max[edge_index]
            # 時間的に過去のクラスタから未来のクラスタへ向かうエッジのみを残す
            edge_index = edge_index[:, t_dst > t_src]

        if self.aggr == 'max':
            # maxプーリング: クラスタ内の特徴量の各次元で最大値をとる
            x, argmax = torch_scatter.scatter_max(data.x, cluster, dim=0)
        else:
            # meanプーリング: クラスタ内の特徴量の平均をとる
            x = _avg_pool_x(cluster, data.x)

        # プーリング後の情報で新しいBatchオブジェクトを作成
        new_data = Batch(batch=batch, x=x, edge_index=edge_index, pos=pos)

        # 元のデータの高さ・幅情報があれば引き継ぐ
        if hasattr(data, "height"):
            new_data.height = data.height
            new_data.width = data.width

        # 新しいノードのxy座標を、ボクセルのグリッドに合わせて丸める
        new_data.pos[:,:2] = self.round_to_pixel(new_data.pos[:,:2], wh_inv=self.wh_inv)

        # transform関数が指定されていれば、それを適用（主にエッジ属性の計算）
        if self.transform is not None:
            if new_data.edge_index.numel() > 0:
                new_data = self.transform(new_data)
            else:
                # エッジが存在しない場合は、空のエッジ属性テンソルを作成
                new_data.edge_attr = torch.zeros(size=(0,pos.shape[1]), dtype=pos.dtype, device=pos.device)

        # バッチ正規化レイヤーが定義されていれば適用
        if self.bn is not None:
            new_data = self.bn(new_data)

        # プーリング処理が完了した新しいグラフデータを返す
        return new_data