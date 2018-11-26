def create_model(opt):
    model = None
    print('Loading model %s...' % opt.model)

    if opt.model == 'fashion_gan':
        from .fashion_gan_model import FashionGANModel
        model = FashionGANModel()
    elif opt.model == 'new_fashion_gan':
        from .new_fashion_gan_model import NewFashionGANModel
        model = NewFashionGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
