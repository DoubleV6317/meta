import os
from .utils import DatasetBase
from .oxford_pets import OxfordPets


template = ['This is a land use image of a {}']
base_classes = ['sparseresidential', 'buildings', 'agricultural', 'chaparral', 'harbor', 'freeway', 'tenniscourt',
                'parkinglot', 'storagetanks', 'river', 'runway', 'beach', 'overpass', 'intersection', 'airplane' ,'denseresidential']

novel_classes = ['golfcourse', 'mobilehomepark', 'baseballdiamond', 'mediumresidential', 'forest', ]


class ClipRs(DatasetBase):


    dataset_dir = 'clip_rs'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'Images')
        self.split_path = os.path.join(self.dataset_dir, 'clip_rs1.json')

        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        few_shot_base = []
        for item in train:
            if item.classname in base_classes:
                few_shot_base.append(item)
        few_shot_base = self.generate_fewshot_dataset(few_shot_base, num_shots=num_shots)
        few_shot_full = self.generate_fewshot_dataset(val, num_shots=num_shots)

        test_novel = []
        for item in test:
            if item.classname in novel_classes:
                test_novel.append(item)
        test_novel = self.generate_fewshot_dataset(test_novel, num_shots=num_shots)

        super().__init__(train=few_shot_base, full=few_shot_full, val=test_novel)
