from .vision import vision_model_map
from .acoustic import acoustic_model_map
from .motion import motion_model_map
from .wireless import wireless_model_map

model_map = {
    'vision': vision_model_map, 
    'acoustic': acoustic_model_map, 
    'motion': motion_model_map,
    'wireless': wireless_model_map,
}