def create_model(opt):
    model = None
    if opt.model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.model == 'poselstm':
        from .poselstm_model import PoseLSTModel
        model = PoseLSTModel()
    else: # cloudnet
        from .cloudnet_model import CloudNetModel
        model = CloudNetModel()
    model.initialize(opt)
    print("Model [%s] was created" % (model.name()))
    return model
