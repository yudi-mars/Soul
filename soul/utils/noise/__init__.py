import motion_noise as mn
import wireless_noise as wn
import vision_noise as vn
import acoustic_noise as an

noising_map = {
    'wireless': {
        'gaussian': wn.add_gaussian_noise,
        'dropout': wn.add_dropouts_noise,
        'impulse': wn.add_impulse_noise,
    },
    'acoustic': {
        'gaussian': an.add_gaussian_noise,
        'dropout': an.add_dropouts_noise,
        'impulse': an.add_impulse_noise,
    },
    'motion': {
        'dropout': mn.add_dropouts_noise,
        'gaussian': mn.add_gaussian_noise,
        'impulse': mn.add_impulse_noised,
    },
    'vision': {
        'gaussian': vn.add_gaussian_noise,
        'impluse': vn.add_impulse_noise,
        'dropout': vn.add_dropouts_noise,
    }
}


