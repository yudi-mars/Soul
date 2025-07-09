class MotionData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []

    input_shape = None, None, None
    num_classes = None

    data_source = None

    def __init__(self, data_dir, T):
        self.data_dir = data_dir
        self.T = T

    def download_data(self):
        raise NotImplementedError