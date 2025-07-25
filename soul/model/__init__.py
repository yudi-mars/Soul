from .vision import vision_model_map
from .acoustic import acoustic_model_map
from .motion import motion_model_map

model_map = {
    **vision_model_map, 
    **acoustic_model_map, 
    **motion_model_map,
}