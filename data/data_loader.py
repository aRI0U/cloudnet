def create_data_loader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader(opt)
    print(data_loader.name())
    return data_loader
