import os
from .utils import DatasetBase
from .oxford_pets import OxfordPets
import nltk
nltk.data.path.append('/mnt/nltk_data')
from nltk.corpus import wordnet as wn

template = ['This is a satellite image of a {}']
base_classes = ['Church', 'Stadium', 'RailwayStation', 'Farmland', 'BaseballField',
                'StorageTanks', 'Beach', 'Parking',
                'Park', 'River', 'School',
                'BareLand',  'Playground', 'Desert','Pond',
                 'Square',  'Mountain', 'Bridge', 'Port', 'Meadow', 'Commercial', 'Center',
                'Resort',
                ]

novel_classes = ['Viaduct','Airport','MediumResidential','DenseResidential','Forest','Industrial', 'SparseResidential',]


attribute_descriptors = {
    "Church": "Tall steeple with a cross, intricate stained glass windows, and arched entrances",
    "Stadium": "Large oval or circular structure with tiered seating surrounding a central field",
    "RailwayStation": "Multiple train tracks converging at platforms with overhead canopies and waiting areas",
    "Farmland": "Expansive fields with visible crop rows, occasional barns, and scattered farm equipment",
    "BaseballField": "Diamond-shaped field with bases, a pitcher's mound, and surrounding bleachers",
    "StorageTanks": "Cylindrical or spherical tanks, often grouped together, with industrial piping and safety railings",
    "Beach": "Sandy shoreline meeting the water, with waves, beach umbrellas, and sunbathers",
    "Parking": "Flat asphalt or multi-level concrete structure with marked parking spaces and vehicles",
    "Airport": "Runways and taxiways with planes, a control tower, and large terminal buildings",
    "Park": "Green open spaces with walking paths, trees, benches, and playgrounds",
    "River": "Flowing water with visible banks, bridges, and possibly boats or wildlife",
    "School": "Building complex with classrooms, playgrounds, sports fields, and parking areas",
    "SparseResidential": "Scattered houses with yards, driveways, and open spaces between properties",
    "BareLand": "Expanses of exposed soil or sand with minimal vegetation and occasional construction equipment",
    "Playground": "Colorful play structures, swings, slides, and soft ground surfaces",
    "MediumResidential": "Clusters of townhouses or apartment buildings with shared outdoor spaces and parking",
    "Square": "Open public space with paved walkways, fountains, statues, and surrounding buildings",
    "Viaduct": "Elevated road or railway bridge with arches or pillars, spanning valleys or other obstacles",
    "Mountain": "High, rugged peaks with rocky slopes, often covered in snow or dense vegetation",
    "Bridge": "Structure spanning water or roads, with visible supports and sometimes suspension cables",
    "Port": "Docks with large ships, cranes, warehouses, and container stacks",
    "Meadow": "Open grassland with wildflowers, gently rolling hills, and occasional trees",
    "Commercial": "Collection of shops and businesses with large storefronts, signs, and parking areas",
    "Center": "Vibrant hub of commerce and culture, bustling with businesses, shops, restaurants, and entertainment venues, often characterized by skyscrapers, busy streets, and diverse population",
    "Resort": "Luxurious buildings with swimming pools, landscaped gardens, and recreational facilities",
    "DenseResidential": "High-density housing with tall apartment buildings and limited green spaces",
    "Desert": "Vast expanse of sand dunes, minimal vegetation, and extreme temperature variations",
    "Forest": "Thick canopy of trees, diverse wildlife, and natural trails",
    "Industrial": "Factories, warehouses, and industrial equipment with chimneys and storage yards",
    "Pond": "Small body of water surrounded by vegetation, often with ducks or other aquatic life"
}

analogous_categories = {'Church': 'Chapel',
    'Stadium': 'Arena',
    'RailwayStation': 'TrainStation',
    'Farmland': 'AgriculturalLand',
    'BaseballField': 'Ballpark',
    'StorageTanks': 'Reservoirs',
    'Beach': 'Seashore',
    'Parking': 'CarPark',
    'Park': 'Garden',
    'River': 'Stream',
    'School': 'EducationalInstitution',
    'BareLand': 'BarrenLand',
    'Playground': 'PlayArea',
    'Desert': 'AridRegion',
    'Pond': 'Waterbody',
    'Square': 'Plaza',
    'Mountain': 'Hill',
    'Bridge': 'Overpass',
    'Port': 'Harbor',
    'Meadow': 'Grassland',
    'Commercial': 'BusinessDistrict',
    'Center': 'Hub',
    'Resort': 'Retreat',
    'Viaduct': 'ElevatedRoadway',
    'Airport': 'Airfield',
    'MediumResidential': 'SuburbanArea',
    'DenseResidential': 'UrbanArea',
    'Forest': 'Woodland',
    'Industrial': 'FactoryArea',
    'SparseResidential': 'RuralArea'}


class Aid(DatasetBase):

    dataset_dir = 'aid'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'AID')
        self.split_path = os.path.join(self.dataset_dir, 'aid_split.json')
        texts = self.generate_texts(base_classes + novel_classes, attribute_descriptors, analogous_categories)

        #self.template = texts['analogous_class_based']
        self.template = template

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        self.save_datasets_to_txt("/mnt/data/aid/aid_all.txt", train, val, test)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        all = train + test + val
        few_shot_base = []
        for item in all:
            if item.classname in base_classes:
                few_shot_base.append(item)
        #few_shot_base = self.generate_fewshot_dataset(few_shot_base, num_shots=num_shots)
        few_shot_full = self.generate_fewshot_dataset(val, num_shots=num_shots)

        test_novel_all = []
        for item in all:
            if item.classname in novel_classes:
                test_novel_all.append(item)
        test_novel = self.generate_fewshot_dataset(test_novel_all, num_shots=num_shots)

        #few_shot_base = few_shot_base + test_novel
        test_novel = list(set(test_novel_all) - set(test_novel))

        self.save_datasets_to_txt("/mnt/data/aid/aid_shot.txt",few_shot_base,few_shot_full,test_novel)
        super().__init__(train=few_shot_base, full=few_shot_full, val=test_novel)

    def generate_texts(self, dataset_classes, attribute_descriptors, analogous_categories):
        texts = {}

        # (1) Category Name-based Texts
        texts['category_name_based'] = [f"a photo of {cls}" for cls in dataset_classes]

        # (2) Attribute-based Texts
        attribute_based_texts = [f"{cls} which has {attribute_descriptors[cls]}" for cls in dataset_classes]
        texts['attribute_based'] = attribute_based_texts

        # (3) Analogous Class-based Texts
        analogous_class_texts = {cls:f"a photo of {cls} which was similar to {analogous_categories[cls]}" for cls in dataset_classes}
        texts['analogous_class_based'] = analogous_class_texts

        # (4) Synonym-based Texts
        synonym_texts = []
        for cls in dataset_classes:
            synonyms = wn.synsets(cls)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                synonym_texts.append(f"a photo of {synonym}")
        texts['synonym_based'] = synonym_texts

        return texts
    def save_datasets_to_txt(self, file_path, few_shot_base, few_shot_full, test_novel):
        with open(file_path, 'w') as file:
            # 写入 few_shot_base 数据集
            file.write("few_shot_base:\n")
            for item in few_shot_base:
                file.write(f"{item.impath}\n")
            file.write("\n")  # 空行分隔

            # 写入 few_shot_full 数据集
            file.write("few_shot_full:\n")
            for item in few_shot_full:
                file.write(f"{item.impath}\n")
            file.write("\n")  # 空行分隔

            # 写入 test_novel 数据集
            file.write("test_novel:\n")
            for item in test_novel:
                file.write(f"{item.impath}\n")
            file.write("\n")  # 空行分隔