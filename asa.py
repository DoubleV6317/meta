import os
from joblib import dump

# 数据
data = [
    ['industrial park', 'factory', 'warehouse district', 'port area'],
    ['badminton court', 'basketball court', 'volleyball court', 'sports field'],
    ['castle', 'manor', 'mansion', 'government building'],
    ['grassland', 'wasteland', 'forest edge', 'low shrubland'],
    ['RV park', 'campground', 'trailer park', 'prefabricated home area'],
    ['cruise ship', 'fishing boat', 'barge', 'warship'],
    ['sports arena', 'coliseum', 'indoor sports hall', 'athletic field'],
    ['hill', 'highland', 'mountain range', 'cliff'],
    ['cropland', 'ranch', 'orchard', 'cultivated field'],
    ['highway', 'expressway', 'main road', 'overpass'],
    ['traffic circle', 'rotary', 'round intersection', 'traffic island'],
    ['iceberg', 'snow-capped peak', 'glacier', 'snowy mountain'],
    ['grassland', 'pasture', 'flower field', 'plain'],
    ['rice terrace', 'balcony', 'platform', 'steppe'],
    ['low-density residential area', 'rural housing', 'small neighborhood', 'dispersed housing'],
    ['airstrip', 'airport runway', 'takeoff strip', 'landing strip'],
    ['peninsula', 'islet', 'coral reef', 'cay'],
    ['golf range', 'practice field', 'country club', 'leisure sports field'],
    ['airfield', 'helicopter', 'drone', 'jet plane'],
    ['dune', 'arid land', 'sand hills', 'barren area'],
    ['train station', 'subway station', 'light rail station', 'transit station'],
    ['pack ice', 'ice floe', 'frozen sea', 'polar ice'],
    ['chapel', 'cathedral', 'basilica', 'monastery'],
    ['railroad track', 'light rail', 'subway line', 'train route'],
    ['cloud layer', 'fog', 'mist', 'vapor'],
    ['marsh', 'swamp', 'wetland reserve', 'reed bed'],
    ['business district', 'shopping mall', 'city center', 'commercial street'],
    ['reservoir', 'lagoon', 'pond', 'tarn'],
    ['silo', 'oil tank', 'water tank', 'container'],
    ['seashore', 'sandy shore', 'coastline', 'seaside'],
    ['pedestrian bridge', 'flyover', 'viaduct', 'skywalk'],
    ['overpass', 'suspension bridge', 'arch bridge', 'viaduct'],
    ['power plant', 'power station', 'coal-fired power plant', 'energy facility'],
    ['baseball field', 'softball field', 'sports field', 'ballpark'],
    ['air terminal', 'aviation hub', 'airfield', 'aerodrome'],
    ['basketball court', 'tennis court', 'badminton court', 'indoor sports hall'],
    ['pivot irrigation field', 'circular crop field', 'agricultural zone', 'farming area'],
    ['high-density housing', 'urban residential area', 'apartment complex', 'neighborhood'],
    ['medium-density housing', 'residential district', 'housing area', 'community'],
    ['woodland', 'woods', 'forest park', 'nature reserve'],
    ['sports ground', 'track and field stadium', 'athletic field', 'school field'],
    ['crossroad', 'junction', 'interchange', 'traffic hub'],
    ['car park', 'parking garage', 'parking area', 'open-air parking'],
    ['stream', 'watercourse', 'creek', 'riverbed']
]

# 指定保存文件路径
file_path = '/mnt/data/nwpu.pkl'

# 保存为pkl文件
dump(data, file_path)
