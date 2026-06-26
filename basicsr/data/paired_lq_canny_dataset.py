"""Paired GT / LQ / Canny from pre-exported folders (codec + HPCM_Base)."""

from pathlib import Path

from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(suffix='basicsr')
class PairedLQCannyDataset(data.Dataset):
    """Load aligned HQ / LQ / Canny triplets with matching filenames."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = Path(opt['gt_path'])
        self.lq_folder = Path(opt['lq_path'])
        self.canny_folder = Path(opt['canny_path'])
        ext = opt.get('image_type', 'png')
        self.paths = []
        for gt_path in sorted(self.gt_folder.glob(f'*.{ext}')):
            lq_path = self.lq_folder / gt_path.name
            canny_path = self.canny_folder / gt_path.name
            if lq_path.is_file() and canny_path.is_file():
                self.paths.append(
                    {
                        'gt_path': str(gt_path),
                        'lq_path': str(lq_path),
                        'canny_path': str(canny_path),
                    }
                )
        if len(self.paths) == 0:
            for gt_path in sorted(self.gt_folder.iterdir()):
                if not gt_path.is_file():
                    continue
                lq_path = self.lq_folder / gt_path.name
                canny_path = self.canny_folder / gt_path.name
                if lq_path.is_file() and canny_path.is_file():
                    self.paths.append(
                        {
                            'gt_path': str(gt_path),
                            'lq_path': str(lq_path),
                            'canny_path': str(canny_path),
                        }
                    )
        if 'max_num' in opt:
            self.paths = self.paths[: int(opt['max_num'])]
        if len(self.paths) == 0:
            raise RuntimeError(
                f'No matched triplets under gt={self.gt_folder}, lq={self.lq_folder}, canny={self.canny_folder}'
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        rec = self.paths[index]
        img_gt = imfrombytes(self.file_client.get(rec['gt_path'], 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(rec['lq_path'], 'lq'), float32=True)
        img_canny = imfrombytes(self.file_client.get(rec['canny_path'], 'canny'), float32=True)
        if self.opt.get('use_hflip', False):
            img_gt, img_lq, img_canny = augment(
                [img_gt, img_lq, img_canny], True, self.opt.get('use_rot', False)
            )
        img_gt, img_lq, img_canny = img2tensor([img_gt, img_lq, img_canny], bgr2rgb=True, float32=True)
        return {
            'gt': img_gt,
            'lq': img_lq,
            'canny': img_canny,
            'gt_path': rec['gt_path'],
            'lq_path': rec['lq_path'],
            'canny_path': rec['canny_path'],
        }

    def __len__(self):
        return len(self.paths)
