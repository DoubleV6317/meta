import os
from .utils import DatasetBase
from .oxford_pets import OxfordPets
import nltk
nltk.data.path.append('/mnt/nltk_data')
from nltk.corpus import wordnet as wn

template = ['This is a land use image of a {}']
base_classes = ['industrial_area', 'tennis_court', 'palace', 'chaparral', 'mobile_home_park',
                'ship',  'stadium', 'mountain', 'rectangular_farmland',  'freeway',
                'roundabout', 'snowberg', 'meadow', 'terrace','sparse_residential',
                'runway',  'island', 'golf_course','airplane','desert',
                'railway_station', 'sea_ice',  'church', 'railway',
                'cloud', 'wetland', 'commercial_area', 'lake', 'storage_tank',
                 'beach', 'overpass', 'bridge', 'thermal_power_station','baseball_diamond',
                ]

novel_classes = ['airport', 'basketball_court', 'circular_farmland','dense_residential','medium_residential',
                 'forest','ground_track_field','intersection','parking_lot', 'river',
                 ]

analogous_categories = {
    "industrial_area": "factory complex, manufacturing zone, industrial park",
    "tennis_court": "badminton court, squash court, pickleball court",
    "palace": "castle, mansion, royal residence",
    "chaparral": "scrubland, shrubland, heathland",
    "mobile_home_park": "trailer park, RV park, campground",
    "ship": "boat, vessel, yacht",
    "stadium": "arena, coliseum, sports complex",
    "mountain": "hill, peak, summit",
    "rectangular_farmland": "crop field, agricultural plot, farmland",
    "freeway": "highway, expressway, interstate",
    "roundabout": "traffic circle, rotary, round circle",
    "snowberg": "glacier, ice cap, snowfield",
    "river": "stream, creek, waterway",
    "meadow": "grassland, prairie, savanna",
    "terrace": "patio, deck, balcony",
    "runway": "airstrip, landing strip, tarmac",
    "ground_track_field": "athletics field, track and field, sports track",
    "island": "isle, atoll, cay",
    "golf_course": "driving range, putting green, golf club",
    "airplane": "jet, aircraft, aeroplane",
    "railway_station": "train station, subway station, transit hub",
    "sea_ice": "ice floe, ice sheet, polar ice",
    "parking_lot": "garage, car park, parking area",
    "church": "synagogue, temple, mosque",
    "railway": "train tracks, rail line, railroad",
    "forest": "woodland, jungle, wilderness",
    "wetland": "marsh, swamp, bog",
    "commercial_area": "business district, downtown, financial center",
    "lake": "pond, reservoir, lagoon",
    "storage_tank": "silos, reservoirs, tanks",
    "beach": "coastline, shore, seaside",
    "overpass": "bridge, viaduct, flyover",
    "bridge": "overpass, viaduct, footbridge",
    "thermal_power_station": "power plant, generating station, power facility",
    "baseball_diamond": "softball field, cricket field, sports field",
    "airport": "airfield, airstrip, heliport",
    "basketball_court": "volleyball court, handball court, sports court",
    "circular_farmland": "pivot irrigation field, circular crop field, round farm plot",
    "cloud": "cumulus, stratus, cirrus",
    "dense_residential": "city, metropolis, urban area",
    "desert": "dunes, wasteland, arid region",
    "harbor": "port, marina, dockyard",
    "intersection": "crossroad, junction, crossing",
    "medium_residential": "town, neighborhood, urban community",
    "sparse_residential": "suburb, village, rural community"
}


attribute_descriptors = {
    "industrial_area": "Cluster of large factories, warehouses, and smokestacks with industrial machinery and vehicles",
    "tennis_court": "Flat rectangular surface with a net in the middle, surrounded by boundary lines and often enclosed by fences",
    "palace": "Grand and ornate building with extensive gardens, courtyards, and elaborate architectural details",
    "chaparral": "Dense, shrubby vegetation on hilly or mountainous terrain, often with visible dry, rocky soil",
    "mobile_home_park": "Rows of mobile homes or trailers with small yards and communal facilities, arranged in a grid-like pattern",
    "ship": "Large vessel with a visible deck, masts or smokestacks, and often docked at a port or sailing on water",
    "stadium": "Massive structure with tiered seating surrounding a central field or arena, often with floodlights and large screens",
    "mountain": "High, rugged peaks with rocky slopes, often covered in snow or dense vegetation",
    "rectangular_farmland": "Large, rectangular plots of cultivated land with visible crop rows and irrigation systems",
    "freeway": "Wide, multi-lane road with overpasses, on-ramps, and heavy traffic",
    "roundabout": "Circular intersection with traffic moving around a central island, often with multiple exits",
    "snowberg": "Massive, floating ice formation with jagged edges and varying shades of white and blue",
    "river": "Flowing body of water with visible banks, often winding through landscapes and sometimes bridged",
    "meadow": "Open field with lush grass, wildflowers, and occasional trees or shrubs",
    "terrace": "Flat, paved area adjacent to a building, often with outdoor furniture and decorative plants",
    "runway": "Long, flat strip of tarmac or concrete at an airport, marked with white lines and surrounded by safety lights",
    "ground_track_field": "Large field with an oval or circular track for running, often with grass in the center and bleachers nearby",
    "island": "Landmass surrounded by water, often with sandy beaches, rocky shores, and vegetation",
    "golf_course": "Expansive, landscaped area with rolling greens, fairways, sand traps, and water hazards",
    "airplane": "Winged aircraft with visible engines, fuselage, and tail, often seen on a runway or in flight",
    "railway_station": "Building with platforms, tracks, waiting areas, and often a central hall with ticket counters",
    "sea_ice": "Expansive, flat sheets of ice floating on the ocean, often with visible cracks and pressure ridges",
    "parking_lot": "Large, paved area with marked spaces for vehicles, often adjacent to buildings",
    "church": "Building with a steeple or bell tower, stained glass windows, and a large interior space with pews",
    "railway": "Parallel tracks with wooden or concrete ties, often surrounded by gravel and leading into the distance",
    "forest": "Dense collection of trees with a thick canopy, underbrush, and various wildlife",
    "wetland": "Marshy area with standing water, tall grasses, and diverse aquatic plants and animals",
    "commercial_area": "Urban zone with numerous shops, businesses, and office buildings, often with heavy pedestrian traffic",
    "lake": "Large body of water surrounded by land, often with visible shorelines and varying water levels",
    "storage_tank": "Large cylindrical or spherical containers, often grouped together, with visible piping and safety features",
    "beach": "Sandy or pebbly shore meeting the water, often with waves, beachgoers, and umbrellas",
    "overpass": "Elevated roadway or bridge crossing over another road or railway, supported by pillars",
    "bridge": "Structure spanning water or roads, with visible supports and sometimes suspension cables",
    "thermal_power_station": "Industrial facility with large cooling towers, chimneys, and extensive piping and machinery",
    "baseball_diamond": "Diamond-shaped field with bases, a pitcher's mound, and surrounding bleachers",
    "airport": "Complex with runways, taxiways, terminals, and control towers, accommodating various aircraft",
    "basketball_court": "Flat rectangular surface with hoops at either end, marked with boundary lines and often indoors",
    "circular_farmland": "Round agricultural plots, often with pivot irrigation systems creating circular crop patterns",
    "cloud": "Mass of condensed water vapor in the sky, varying in shape, size, and color",
    "dense_residential": "High-density housing with tall apartment buildings and limited green spaces",
    "desert": "Vast expanse of sand dunes, minimal vegetation, and extreme temperature variations",
    "harbor": "Docks with large ships, cranes, warehouses, and container stacks",
    "intersection": "Junction where multiple roads meet, often controlled by traffic lights or signs",
    "medium_residential": "Clusters of townhouses or apartment buildings with shared outdoor spaces and parking",
    "sparse_residential": "Scattered houses with yards, driveways, and open spaces between properties"
}


class Nwpu(DatasetBase):


    dataset_dir = 'nwpu'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'NWPU')
        self.split_path = os.path.join(self.dataset_dir, 'NWPU_try.json')

        texts = self.generate_texts(base_classes + novel_classes, attribute_descriptors, analogous_categories)

        #self.template = texts['analogous_class_based']
        self.template = template
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        few_shot_base = []
        all = train + test + val
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
        test_novel = list(set(test_novel_all) - set(test_novel))
        super().__init__(train=train, full=val, val=test)

    @staticmethod
    def generate_texts(dataset_classes, attribute_descriptors, analogous_categories):
        texts = {}

        # (1) Category Name-based Texts
        texts['category_name_based'] = [f"a photo of {cls}" for cls in dataset_classes]

        # (2) Attribute-based Texts
        if attribute_descriptors:
            attribute_based_texts = [f"{cls} which has {attribute_descriptors[cls]}" for cls in dataset_classes]
            texts['attribute_based'] = attribute_based_texts

        # (3) Analogous Class-based Texts
        if analogous_categories:
            analogous_class_texts = [(cls, f"a {cls} similar to {analogous_categories[cls]}") for cls in dataset_classes]
            texts['analogous_class_based'] = analogous_class_texts

        # (4) Synonym-based Texts
        synonym_texts = []
        for cls in dataset_classes:
            synonyms = wn.synsets(cls)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()
                synonym_texts.append((cls, f"a photo of {synonym}"))
        texts['synonym_based'] = synonym_texts

        return texts