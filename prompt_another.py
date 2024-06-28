import nltk
nltk.data.path.append('/mnt/nltk_data')
from nltk.corpus import wordnet as wn

# 手动提供的属性描述符和类比分类
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

analogous_categories = {
    "Church": "Synagogue, temple, mosque",
    "Stadium": "Arena, coliseum, sports complex",
    "RailwayStation": "Bus terminal, subway station, transit hub",
    "Farmland": "Ranch, plantation, vineyard",
    "BaseballField": "Soccer field, basketball court, tennis court",
    "StorageTanks": "Silos, reservoirs, tanks",
    "Beach": "Coastline, shore, seaside",
    "Parking": "Garage, car park, parking lot",
    "Airport": "Airfield, airstrip, heliport",
    "Park": "Garden, botanical garden, nature reserve",
    "River": "Stream, creek, waterway",
    "School": "University, college, academy",
    "SparseResidential": "Suburb, village, rural community",
    "BareLand": "Wasteland, desert, barren terrain",
    "Playground": "Amusement park, recreation center, play area",
    "MediumResidential": "Town, neighborhood, urban community",
    "Square": "Plaza, market square, town square",
    "Viaduct": "Overpass, bridge, aqueduct",
    "Mountain": "Hill, peak, summit",
    "Bridge": "Overpass, viaduct, footbridge",
    "Port": "Harbor, marina, dockyard",
    "Meadow": "Grassland, prairie, savanna",
    "Commercial": "Business district, downtown, financial center",
    "Center": "Downtown, city center, central district",
    "Resort": "Spa, retreat, vacation destination",
    "DenseResidential": "City, metropolis, urban area",
    "Desert": "Dunes, wasteland, arid region",
    "Forest": "Woodland, jungle, wilderness",
    "Industrial": "Factory, manufacturing plant, industrial zone",
    "Pond": "Lake, pool, reservoir"
}

one_v_one_descriptors = {
    ("cat", "dog"): "Cats have retractable claws and tend to be more independent, whereas dogs have non-retractable claws and are generally more social.",
    ("cat", "bird"): "Cats have fur and move on four legs, while birds have feathers and can fly.",
    ("dog", "bird"): "Dogs have fur and are land animals, while birds have feathers and can fly."
}

def generate_texts(dataset_classes, attribute_descriptors, analogous_categories, one_v_one_descriptors):
    texts = {}

    # (1) Category Name-based Texts
    texts['category_name_based'] = [f"a photo of {cls}" for cls in dataset_classes]

    # (2) Attribute-based Texts
    attribute_based_texts = [f"{cls} which has {attribute_descriptors[cls]}" for cls in dataset_classes]
    texts['attribute_based'] = attribute_based_texts

    # (3) Analogous Class-based Texts
    analogous_class_texts = [f"a {cls} similar to {analogous_categories[cls]}" for cls in dataset_classes]
    texts['analogous_class_based'] = analogous_class_texts

    # (4) Synonym-based Texts
    synonym_texts = []
    for cls in dataset_classes:
        synonyms = get_synonyms(cls)
        if synonyms:
            #synonym = synonyms[0].lemmas()[0].name()
            synonym_texts.append(f"a photo of {synonyms}")
    texts['synonym_based'] = synonym_texts

    # (5) 1v1-based Texts
    one_v_one_texts = []
    for (class1, class2), descriptor in one_v_one_descriptors.items():
        one_v_one_texts.append(f"Because of {descriptor}, {class1} is different from {class2}")
        one_v_one_texts.append(f"{class1} can be distinguished from {class2} by the characteristics of {descriptor}")
    texts['one_v_one_based'] = one_v_one_texts

    return texts
def get_synonyms(word):
    # 使用WordNet查找同义词
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)
# Example usage
dataset_classes = [ 'Church', 'Stadium', 'RailwayStation', 'Farmland', 'BaseballField',
                    'StorageTanks', 'Beach', 'Parking', 'Airport',
                    'Park', 'River', 'School', 'SparseResidential',
                    'BareLand',  'Playground', 'MediumResidential',
                    'Square', 'Viaduct', 'Mountain', 'Bridge', 'Port', 'Meadow', 'Commercial', 'Center',
                    'Resort','DenseResidential','Desert','Forest','Industrial', 'Pond']
texts = generate_texts(dataset_classes, attribute_descriptors, analogous_categories, one_v_one_descriptors)
for key, value in texts.items():
    print(f"{key}: {value}")
