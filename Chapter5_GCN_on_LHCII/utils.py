    torch.manual_seed = 1
    best_epoch_num = 0
    best_loss_valid = 1e10
    earlystopping = True
    patient = 50
    PATH = './gcn_net_trained_9_16batched_MSELoss.pth'


    batch_size = 16
    max_epochs = 30
    DATA_PATH = "./NLLZ_7_PROD_1_frames/"

    dataset_size = len(My_dataset(DATA_PATH))
    perm = torch.randperm(dataset_size).numpy()
    partition = {}
    partition["train"] = perm[:int(dataset_size*8/10)]
    partition["validation"] = perm[int(dataset_size*8/10):int(dataset_size*9/10)]
    partition["test"] = perm[int(dataset_size*9/10):]
    partition["total"] = perm[range(int(dataset_size))]

    train_sampler = SubsetRandomSampler(partition["train"])
    validation_sampler = SubsetRandomSampler(partition["validation"])
    test_sampler = SubsetRandomSampler(partition["test"])
    total_sampler = SubsetRandomSampler(partition["total"])
    
    train_loader = DataLoader(My_dataset(DATA_PATH), 
                              batch_size=batch_size, 
                              sampler=train_sampler,
                              shuffle=False, 
                              num_workers=8)

    validation_loader = DataLoader(My_dataset(DATA_PATH), 
                              batch_size=batch_size, 
                              sampler=validation_sampler,
                              shuffle=False, 
                              num_workers=8)

    test_loader = DataLoader(My_dataset(DATA_PATH), 
                              batch_size=batch_size, 
                              sampler=test_sampler,
                              shuffle=False, 
                              num_workers=8)
    
    total_loader = DataLoader(My_dataset(DATA_PATH), 
                              batch_size=batch_size, 
                              sampler=total_sampler,
                              shuffle=False, 
                              num_workers=8)