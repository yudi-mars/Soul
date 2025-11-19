import motion_noise as mn
import wireless_noise as wn
import vision_noise as vn
import acoustic_noise as an

noising_map = {
    'wireless': {
        'gaussian': wn.add_gaussian_noise,
        'dropout': wn.add_dropouts_noise,
    },
    'acoustic': {
        'gaussian': an.add_gaussian_noise,
        'dropout': an.add_dropouts_noise,
    },
    'motion': {
        'gaussian': mn.add_gaussian_noise,
        'dropout': mn.add_dropouts_noise,
    },
    'vision': {
        'gaussian': vn.add_gaussian_noise,
        'dropout': vn.add_dropouts_noise,
    }
}


