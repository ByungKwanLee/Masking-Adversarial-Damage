class MaskParameterGenerator(object):
    def __init__(self, model):
        self.model = model

    def mask_parameters(self):
        for name, param in self.model.named_parameters():
            if ('mask' in name) and (param!=None):
                yield param

    def non_mask_parameters(self):
        for name, param in self.model.named_parameters():
            if not 'mask' in name:
                yield param
