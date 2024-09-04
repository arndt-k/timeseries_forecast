import torch
import numpy as np
from torch.nn import Linear, Dropout, ReLU
from tqdm import tqdm
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


class TSTrafo(torch.nn.Module):
    def __init__(self,pred_length, input_size,lags_seq, num_time_feat, 
                 num_static_cat_feat, cardinality, embedding_dim, enc_layers,
                 dec_layers, model_dim):
        """Wrapper for time series prediction transformer

        Args:
            pred_length (int): length of target sequence
            input_size (int): number of variables in main sequence
            lags_seq (list of int): which lagged features to add
            num_time_feat (int): number of ancillary time-related features
            num_static_cat_feat (int): time series-id features (purpose?)
            cardinality (int): number of timeseries (purpose?)
            embedding_dim (int): fancyness of static embeddings
            enc_layers (int): complexity of encoder part
            dec_layers (int): complexity of decoder part
            model_dim (int): dimension of individual layers
        """
        super(TSTrafo,self).__init__()
        self.config = TimeSeriesTransformerConfig(
        prediction_length=pred_length,
        input_size=input_size,
        context_length=pred_length * 2,
        lags_sequence=lags_seq,
        num_time_features=num_time_feat,
        num_static_categorical_features=num_static_cat_feat,
        cardinality=[cardinality],
        embedding_dimension=[embedding_dim],
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        d_model=model_dim,
        scaler=False, #because the dataset scales the data
    )

        self.model = TimeSeriesTransformerForPrediction(self.config)
    
    def forward(self,x):
        time_feat,timestamps,zhl_id, past_val, target = x
        past_time_feat = torch.cat((timestamps[:,:-self.config.prediction_length,...],
                                    time_feat[:,:-self.config.prediction_length]),dim=-1)
        future_time_feat = torch.cat((timestamps[:,-self.config.prediction_length:],
                                      time_feat[:,-self.config.prediction_length:]),dim=-1) 
        past_mask = ~torch.isnan(past_val)
        future_mask = ~torch.isnan(target)
        if not torch.all(past_mask):
            print(torch.sum(~past_mask))
        if not torch.all(future_mask):
            print(torch.sum(~future_mask))
        return self.model(past_values= past_val,
                          past_time_features = past_time_feat,
                          past_observed_mask = past_mask,
                          future_values = target,
                          future_time_features = future_time_feat,
                          future_observed_mask = future_mask,
                          static_categorical_features = zhl_id.type(torch.int64).reshape(-1,1) if self.config.num_static_categorical_features >0 else None,
                          )
    
    def generate(self,x):
        #Function used for inference. Returns a an ensemble of preditions which are averaged
        time_feat,timestamps,zhl_id, past_val, _ = x
        past_time_feat = torch.cat((timestamps[:,:-self.config.prediction_length,...],
                                    time_feat[:,:-self.config.prediction_length]),dim=-1)
        future_time_feat = torch.cat((timestamps[:,-self.config.prediction_length:],
                                      time_feat[:,-self.config.prediction_length:]),dim=-1) 
        return self.model.generate(past_values= past_val,
                          past_time_features = past_time_feat,
                          past_observed_mask = ~torch.isnan(past_val),
                          #future_values = target.reshape(len(target),-1),
                          future_time_features = future_time_feat,
                          #future_observed_mask = [1],
                          static_categorical_features=zhl_id.type(torch.int64).reshape(-1,1) if self.config.num_static_categorical_features >0 else None,
                          ).sequences.mean(dim=1)
    
   


def train(all_models, optims, loss_fn, n_epochs, train_loader,
           test_loader, writer=None,device="cuda:0",
           tags = ["FCNN", "LSTM", "Trafo"]):
    """simultaneously train all models

    Args:
        all_models (list of pytorch models): train them
        optims (list of pytorch optimizers): for each model
        loss_fn (pytorch loss function): to use for models that dont supply own loss
        n_epochs (int): run through trainloader this often
        train_loader (pytorch dataloader): iterates through batches of train data
        test_loader (pytorch dataloader): iterates through batches of test data
        writer (tensorbard summary writer, optional): to log the training. Defaults to None.
        device (str, optional): where to do the computations. Defaults to "cuda:0".
        tags (list, optional): nametags for the models used for logging. Defaults to ["FCNN", "LSTM", "Trafo"].

    Returns:
        testloss: average loss over the test data after the last training epoch
    """
    #progress bar
    pbar_gen = tqdm(range(n_epochs), total = n_epochs*len(train_loader), leave=True)
    #instantiate to have something to log before first test
    testloss = [0 for _ in all_models]
    [x.train() for x in all_models]
    for epoch in pbar_gen:
        avg_loss = [0 for _ in all_models]
        with torch.cuda.device(device):
            for i, data in enumerate(train_loader):
                #failsafe
                if data is None:
                    continue
                #move stuff to the gpu because model is on the gpu
                data = [x.cuda() for x in data]
                *inputs, targets = data
                for j,(model, optim) in enumerate(zip(all_models, optims)):
                    #remove overhang from last iteration
                    optim.zero_grad()
                    # Make predictions for this batch
                    outputs = model(data)
                    
                    # Compute the loss and its gradients
                    # the trafo has its own loss which is not absolutely comparable 
                    # to the losses applicable to the other models
                    try:     
                        loss = outputs.loss
                    except Exception:
                        loss = loss_fn(outputs, targets)
                    loss.backward()
                    avg_loss[j] += loss/len(train_loader)

                    # Adjust learning weights
                    optim.step()

                if i < len(train_loader)-2:#this should only be -1 but doesnt work?
                   pbar_gen.update(1)
            
            #basically same as trainloader it but no gradient/backprop
            # only done every once in a while
            if (epoch%int(n_epochs/4+1)==0 and epoch!=0) or epoch==n_epochs-1:
                testloss=[0 for _ in all_models]
                with torch.no_grad():
                    for i, data in enumerate(test_loader):
                        if data is None:
                            continue
                        data = [x.cuda() for x in data]
                        *inputs, targets = data
                        for j,model in enumerate(all_models):
                            outputs = model(data)
                            try:
                                testloss[j] += outputs.loss/len(test_loader)
                            except Exception:
                                testloss[j] += loss_fn(outputs, targets)/len(test_loader)
                        
                writer.add_scalars('Test Loss',
                    dict(zip(tags,testloss)),
                    (epoch+1)*len(train_loader))
        
        writer.add_scalars('Training Loss',
                        dict(zip(tags,avg_loss)),
                        epoch*len(train_loader)+i/len(train_loader))
        pbar_gen.set_description("Epoch: {} Loss: {:.2E} Testloss: {:.2E}".format(
            epoch+1, min(avg_loss), min(testloss))
        )

    return testloss

def evaluate(model, val_loader, writer=None,device="cuda:0"):
    """evaluates a single model on held out data

    Args:
        model (_type_): _description_
        val_loader (_type_): _description_
        writer (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cuda:0".

    Returns:
        _type_: _description_
    """
    model.eval()
    pbar_gen = tqdm(val_loader, leave=True, total = len(val_loader))
    out = []
    with torch.no_grad():
        with torch.cuda.device(device):
            for data in pbar_gen:
                if data is None:
                    continue
                *inputs, targets = data
                outputs = model.generate(data)
                out.append((inputs,
                            outputs,
                            targets))
    return out

class FCNN(torch.nn.Module):
    def __init__(self,in_dim, out_dim):
        """Stupid 2-layer fully connected network that uses no ancillary data

        Args:
            in_dim (int): length of input sequence
            out_dim (int): lengthe of output sequence
        """
        super(FCNN, self).__init__()
        self.input_net = Linear(in_dim, 500)
        self.output_net = Linear(500, out_dim)
        self.dropout = Dropout(0.2)
        self.relu = ReLU()

    def forward(self,x): 
        _,_, zhl_id,past_val, _ = x
        if len(past_val.shape)==2:
            # if univariate timeseries
            return self.relu(self.output_net(
                self.dropout(
                self.input_net(past_val)))
            )
        else:
            # if multivariate timeseries
            return torch.stack([self.relu(self.output_net(self.dropout(
                                    self.input_net(past_val[...,i])))) 
                                    for i in range(past_val.shape[-1])],dim=-1)
    
    def generate(self,x):
        return self.forward(x)

class LSTM(torch.nn.LSTM):
    def __init__(self,seq_length, lag_list, pred_length,**kwargs):
        """wrapper around an LSTM implementation to fit my dataloader
            same variables as anywhere else
        Args:
            seq_length (int): _description_
            lag_list (list of int): _description_
            pred_length (list of int): _description_
        """
        super(LSTM,self).__init__(proj_size=2,dropout=0.2,**kwargs)
        self.pred_length=pred_length
        self.Linear = Linear(seq_length+max(lag_list),pred_length)

    def forward(self,x):
        time_feat,timestamps,zhl_id, past_val,_ = x
        past_time_feat = torch.cat((timestamps[:,:-self.pred_length,...],
                                    time_feat[:,:-self.pred_length]),dim=-1)
        
        inputs = torch.cat((past_val,
                            past_time_feat),dim=-1)
        #get rid of projection size
        sequence = super(LSTM,self).forward(inputs)[0].squeeze()
        #cut down output sequence to just the relevant timesteps
        return torch.stack([self.Linear(sequence[...,i]) 
                            for i in range(sequence.shape[-1])],dim=-1)
    
    def generate(self,x):
        return self.forward(x)
