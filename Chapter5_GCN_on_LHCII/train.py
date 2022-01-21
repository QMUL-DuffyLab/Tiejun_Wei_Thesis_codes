import numpy as np
import torch
import time

from torch.autograd import Variable
from configs import args
from my_dataset_4cluster_2 import *
from GCN_regression import *



def train_net(model, train_loader, validation_loader, max_epochs, loss_func, prefix, device):
    """
    args::

    model: the GCN net defined
    loader: train_loader, validation_loader or test_loader,
    max_epochs: 100 - 1000 usually enough.
    loss_func: the loss function, e.g. MSE
    PATH: the prefix of the job. e.g. PATH = './gcn_net_trained_9test_16batched_MSELoss.pth'
    device: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') specify the GPU.
    """
    #initalize the list recording loss.
    hist_train_loss = []
    hist_val_loss = []

    model_saving_PATH = './models/' + prefix + ".pt"
    result_saving_PATH = './outputs/' + prefix 



    start = time.time()
    for epoch in range(max_epochs):
        #training here.
        epoch_start = time.time()
        model.train()
        for i, batch in enumerate(train_loader):
            
            #unpack values from loader.
            batch_vec, X, edge_index, edge_weight, y = batch.batch, batch.X, batch.edge_index, batch.edge_attr, batch.y
            
            #transfer to GPU
            #x, edge_index, edge_attr, y = Variable(torch.as_tensor(x)).to(device), Variable(torch.as_tensor(edge_index)).to(device), Variable(torch.as_tensor(edge_attr)).to(device), Variable(torch.as_tensor(y)).to(device)
            
            batch_vec, X, edge_index, edge_weight = batch_vec.to(device), Variable(torch.as_tensor(X, dtype=torch.float32)).to(device), edge_index.to(device), edge_weight.to(device)
            #y = y.to(device)
            y = Variable(torch.as_tensor(y, dtype=torch.float32)).to(device)
            
            #do the model computation here.
            optimizer.zero_grad()
            
            #the input is the x and edge_index. the true label is y
            prediction_train = model(X, edge_index, edge_weight, batch_vec) #forward 
            
            #print("printing y shape here:")
            #print(prediction_train.shape)
            #print(y.shape)
            loss = loss_func(prediction_train, y)/args.batch_size

            loss.backward() # backward
            optimizer.step() # optimizer step
            #optimizer.zero_grad() # don't need it here, redundant.
        
            #print the statistics here;
            print('epoch [%d]; batch No. [%d] loss: %.2f' %(epoch + 1, i + 1, loss))
            
            if i % 5 == 0:
                hist_train_loss.append(loss)
        
        with torch.no_grad():
            for j, batch in enumerate(validation_loader):
                batch_vec_val, X_val, edge_index_val, edge_weight_val, y_val = batch.batch, batch.X.double(), batch.edge_index, batch.edge_attr, batch.y
                batch_vec_val, X_val, edge_index_val, edge_weight_val = batch_vec_val.to(device), Variable(torch.as_tensor(X_val, dtype=torch.float32)).to(device), edge_index_val.to(device), edge_weight_val.to(device)
                #y_val = y_val.to(device)
                y_val = Variable(torch.as_tensor(y_val, dtype=torch.float32)).to(device)
                
                model.eval()
                
                prediction_validation = model(X_val, edge_index_val, edge_weight_val, batch_vec_val)
                loss_validation = loss_func(prediction_validation, y_val)/args.batch_size
                print('epoch [%d]; batch No. [%d] loss_val: %.2f' %(epoch + 1, j + 1, loss_validation))
            
                if j % 2 == 0:
                    hist_val_loss.append(loss_validation)
        """        
        print(
                "[Epoch %d/%d] [Batch %d/%d] [training loss: %f] [validation loss: %f] [time used(sec): %f]]"
                % (epoch, max_epochs, i, len(train_loader), loss.item(), loss_val.item(), (time.time() - start))
            )
        """
       # if (epoch + 1) % 1 ==0 or epoch ==0:
        print(f"epoch {epoch +1}, loss {loss.item():.4f}, loss_val {loss_validation.item():.4f}, time used {(time.time() - epoch_start):.1f}")
        
    
        if args.earlystopping:
                if args.best_loss_valid > loss_validation:
                    args.best_loss_valid = loss_validation
                    args.best_epoch_num = epoch
                    torch.save(model.state_dict(), model_saving_PATH)
                    print('save model')
                if epoch - args.best_epoch_num > args.patient:
                    break
        
    

    torch.save(model.state_dict(), (model_saving_PATH + '.final'))
        

    #save the loss
    np.save((result_saving_PATH + "_train_loss.npy" ), hist_train_loss)
    np.save((result_saving_PATH + "_val_loss.npy" ), hist_val_loss)

    print("training finished:")
    print("total epochs trained: %s" %max_epochs)
    print("total time used: %s" %(time.time() - start))


if __name__ == "__main__":
    
    #earlystopping, best_loss_valid, best_epoch_num, patient = args.earlystopping, args.best_loss_valid, args.best_epoch_num, args.patient
    
    
    #fix the random seed.
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ############################################################################
    #data loader initalization
    #DATA_PATH = './data/'
    dataset_size = 1000
    perm = torch.randperm(dataset_size).numpy()
    partition = {}
    partition["train"] = perm[:int(dataset_size*8/10)]
    partition["validation"] = perm[int(dataset_size*8/10):int(dataset_size*9/10)]
    partition["test"] = perm[int(dataset_size*9/10):]

    train_loader = DataLoader(torch.utils.data.Subset(My_dataset(), partition["train"]), 
                              batch_size=args.batch_size, 
                              #sampler=train_sampler,
                              shuffle=True, 
                              num_workers=8)

    validation_loader = DataLoader(torch.utils.data.Subset(My_dataset(), partition["validation"]), 
                              batch_size=args.batch_size, 
                              #sampler=validation_sampler,
                              shuffle=True, 
                              num_workers=8)

    test_loader = DataLoader(torch.utils.data.Subset(My_dataset(), partition["test"]), 
                              batch_size=args.batch_size, 
                              #sampler=test_sampler,
                              shuffle=True, 
                              num_workers=8)

    ##########################################################################

    model = GCN_regression_attn_Net(n_features = 6, 
                               nhid1 = args.hid1, 
                               nhid2 = args.hid2, 
                               nhid3 = args.hid3,  
                               batch_size = args.batch_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-6, weight_decay=5e-4)
    loss_func = torch.nn.MSELoss()

    train_net(model = model, train_loader = train_loader, validation_loader = validation_loader, max_epochs = args.max_epochs, loss_func = loss_func, prefix = args.prefix, device = device)