from pathlib import Path
from typing import Callable, Optional
from functools import lru_cache

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.data import Dataset

from .dsec_det.dataset import DSECDet
from .dsec_det.directory import BaseDirectory
from .dsec_utils import (
    compute_class_mapping,
    crop_tracks,
    filter_small_bboxes,
    filter_tracks,
    map_classes,
    rescale_tracks,
)
from ..augment import init_transforms
from src.utils.data_utils import to_data


def tracks_to_array(tracks):
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)



def interpolate_tracks(detections_0, detections_1, t):
    assert len(detections_1) == len(detections_0)
    if len(detections_0) == 0:
        return detections_1

    t0 = detections_0['t'][0]
    t1 = detections_1['t'][0]

    assert t0 < t1

    # need to sort detections
    detections_0 = detections_0[detections_0['track_id'].argsort()]
    detections_1 = detections_1[detections_1['track_id'].argsort()]

    r = ( t - t0 ) / ( t1 - t0 )
    detections_out = detections_0.copy()
    for k in 'xywh':
        detections_out[k] = detections_0[k] * (1 - r) + detections_1[k] * r

    return detections_out

class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_file(self):
        return self.root / "left/events_2x.h5"


class DSEC(Dataset):
    MAPPING = dict(pedestrian="pedestrian", rider=None, car="car", bus="car", truck="car", bicycle=None,
                   motorcycle=None, train=None)
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable]=None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False):
        """
        パラメータ:
        - root: データセットのルートディレクトリのパス。
        - split: データセットの分割名（例: "train", "val", "test"）。
        - transform: データに適用する変換関数。
        - debug: デバッグモードのフラグ。デバッグ情報を出力
        - min_bbox_diag: 最小バウンディングボックスの対角線長。これ以下のバウンディングボックスは除外されます。
        - min_bbox_height: 最小バウンディングボックスの高さ。これ以下のバウンディングボックスは除外されます。
        - scale: 画像のスケーリング係数。画像の幅と高さをこの値で割ります。
        - cropped_height: クロップ後の画像の高さ。画像はこの高さに合わせて切り取られます。
        - only_perfect_tracks: 完全なトラックのみを使用するかどうかのフラグ。
        - demo: デモモード。データセットの一部のみを使用します。
        - no_eval: 評価モードを無効にするフラグ。Trueの場合、評価用のデータは使用されません。
        """

        Dataset.__init__(self)

        split_config = None
        if not demo:
            split_config = OmegaConf.load(Path(__file__).parent / "dsec_det" / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        for directory in self.dataset.directories.values():
            directory.events = EventDirectory(directory.events.root)

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.debug = debug
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        self.image_index_pairs, self.track_masks = filter_tracks(dataset=self.dataset, image_width=self.width,
                                                                 image_height=self.height,
                                                                 class_remapping=self.class_remapping,
                                                                 min_bbox_height=min_bbox_height,
                                                                 min_bbox_diag=min_bbox_diag,
                                                                 only_perfect_tracks=only_perfect_tracks,
                                                                 scale=scale)
        
    def set_num_us(self, num_us):
        self.num_us = num_us

    def len(self):
        return sum(len(d) for d in self.image_index_pairs.values())
    
    def get(self, idx):
        dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
        image_index_0, image_index_1 = image_index_pairs[idx]
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]

        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
        detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

        detections_0 = self.preprocess_detections(detections_0)
        detections_1 = self.preprocess_detections(detections_1)

        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        image_0 = self.preprocess_image(image_0)

        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        if self.num_us >= 0:
            image_ts_1 = image_ts_0 + self.num_us
            events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
            if not self.no_eval:
                detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)

        # here, the timestamp of the events is no longer absolute
        events = self.preprocess_events(events)

        # convert to torch geometric data
        data = to_data(**events, bbox=tracks_to_array(detections_1), bbox0=tracks_to_array(detections_0), t0=image_ts_0, t1=image_ts_1,
                       width=self.width, height=self.height, time_window=self.time_window,
                       image=image_0, sequence=str(dataset.root.name))

        if self.transform is not None:
            data = self.transform(data)

        # remove bboxes if they have 0 width or height
        mask = filter_small_bboxes(data.bbox[:, 2], data.bbox[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox = data.bbox[mask]
        mask = filter_small_bboxes(data.bbox0[:, 2], data.bbox0[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox0 = data.bbox0[mask]

        return data

    def preprocess_detections(self, detections):
        # 検出結果の座標をスケールに合わせて調整します。
        detections = rescale_tracks(detections, self.scale)

        # 検出結果を指定された幅(width)と高さ(height)の範囲内に切り取ります（クロッピング）。
        detections = crop_tracks(detections, self.width, self.height)

        # クラスIDを新しいマッピング（class_remapping）に基づいて変換します。
        # 例えば、元々[1, 2, 3, 5]だったクラスIDを[0, 1, 2, 3]のように再割り当てします。
        detections['class_id'], _ = map_classes(detections['class_id'], self.class_remapping)

        return detections

    def preprocess_events(self, events):
        # イベントのy座標が指定された高さ(height)よりも小さいものだけにフィルタリングします。
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}

        # イベントが存在する場合、タイムスタンプを調整します。
        if len(events['t']) > 0:
            # 最新のイベントのタイムスタンプを基準に、全イベントのタイムスタンプを
            # [time_window - 差分, time_window] の範囲に正規化します。
            events['t'] = self.time_window + events['t'] - events['t'][-1]

        # 極性(polarity)の値を変換します。
        # 元の値が {0, 1} の場合、それを {-1, 1} に変換します。
        events['p'] = 2 * events['p'].reshape((-1,1)).astype("int8") - 1

        return events

    def preprocess_image(self, image):
        # 画像を `self.scale * self.height` の高さに切り取ります。
        image = image[:self.scale * self.height]

        # 画像を指定された幅(width)と高さ(height)にリサイズします。
        # cv2.INTER_CUBICは、高品質な補間方法の一つです。
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        # NumPy配列からPyTorchのテンソルに変換し、次元の順番を(H, W, C)から(C, H, W)に入れ替えます。
        # これは多くのPyTorchモデルで標準的な入力形式です。
        image = torch.from_numpy(image).permute(2, 0, 1)

        # バッチ次元を追加します。(C, H, W) -> (1, C, H, W)
        # これにより、画像を1枚のバッチとしてモデルに入力できます。
        image = image.unsqueeze(0)

        return image

    def rel_index(self, index, directory_name=None):
        """
        データセット内のインデックスに対応する情報を取得します。

        directory_nameが指定された場合は、そのディレクトリ内の相対インデックスとして
        扱い、高速に情報を取得します。指定されない場合は、データセット全体の
        通しインデックスとして扱い、対応するサブシーケンスの情報を検索します。

        Args:
            index (int): データインデックス。directory_name指定時はその中での相対インデックス。
            directory_name (str, optional): 対象のディレクトリ名。Defaults to None.

        Raises:
            IndexError: 通しインデックスが範囲外の場合に送出されます。
            KeyError: 指定されたdirectory_nameが存在しない場合に送出されます。

        Returns:
            tuple: 以下の要素を含むタプル
                - directory (str): データが格納されているディレクトリのパス。
                - image_index_pairs (list): 画像インデックスのペア情報。
                - track_mask (Any): 対応するトラックのマスク。
                - relative_index (int): サブシーケンス内での相対インデックス。
        """
        # Case 1: directory_nameが指定されている場合（高速アクセス）
        if directory_name is not None:
            directory = self.dataset.directories[directory_name]
            image_index_pairs = self.image_index_pairs[directory_name]
            track_mask = self.track_masks[directory_name]
            
            # この場合、引数のindexがそのまま相対インデックスとなる
            return directory, image_index_pairs, track_mask, index

        # Case 2: directory_nameが指定されていない場合（通しインデックスで検索）
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            num_items_in_folder = len(image_index_pairs)

            if index < num_items_in_folder:
                directory = self.dataset.directories[name]
                track_mask = self.track_masks[name]
                # このスコープに入った時点でのindexが、そのフォルダ内での相対インデックスとなる
                return directory, image_index_pairs, track_mask, index
            
            # 次のフォルダを調べるためにインデックスを更新
            index -= num_items_in_folder

        raise IndexError("The given index is out of range for the entire dataset.")