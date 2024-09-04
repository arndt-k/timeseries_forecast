# %%
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import glob
from datetime import datetime
from itertools import product

import numpy as np
import hvplot
import polars as pl
import matplotlib.pyplot as plt
import importlib
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler as SRS
from torch.optim import SGD, Adam, AdamW

import models, utils
importlib.reload(models)
importlib.reload(utils)
from models import FCNN, train, evaluate, TSTrafo, LSTM
from utils import  FWData,double_collate,count_params

# %%

if __name__=="__main__":
    print("Cuda?:{}".format( torch.cuda.is_available()))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    name = "test"
    timestamp = str(datetime.now()).replace(":","").replace(" ","")
    writer = SummaryWriter('logs/{}_{}'.format(name, timestamp))
    
    ###Load and Process Data
    path = '/dummy/path/file.csv'
    # TT_TU is outside temperature,
    # v_temp is input temperature
    # r_temp is return temperature 
    dataset = FWData(path,seq_length=48,lag_list=range(2,10), pred_length=24,
                     limit =None,other = ["v_temp", "TT_TU"],scale=True)
    
    min_idx = dataset.seq_length + max(dataset.lag_list)
    splits = (.7, .29, .01)
    timeseries_idx = dataset.zhl
    long_timeseries = dataset.df.group_by(pl.col("Nr")).len().filter(pl.col("len").ge(100)).select(pl.col("Nr")).to_numpy().squeeze()
    assert len(long_timeseries)>0
    timeseries_idx = [x for x in timeseries_idx if x in long_timeseries]
    max_len = max([len(dataset.df.filter(pl.col("Nr")==x)) for x in timeseries_idx])-dataset.pred_length
    timestep_idx = range(min_idx, max_len)
    indices = np.array(list(product(timestep_idx,timeseries_idx)))
    np.random.shuffle(indices)

    with torch.cuda.device(device):
        train_sampler =SRS(indices[:int(len(indices)*splits[0])])
        test_sampler = SRS(indices[int(len(indices)*splits[0]):int(len(indices)*splits[0])+int(len(indices)*splits[1])])
        val_sampler = indices[len(indices)-int(len(indices)*splits[2]):]
        
        trainloader = DataLoader(dataset, sampler=train_sampler,
                                  batch_size=24,collate_fn=double_collate,
                                  num_workers=8, multiprocessing_context="spawn",
                                  )
        testloader = DataLoader(dataset, sampler=test_sampler,
                                 batch_size=24,collate_fn=double_collate,
                                  num_workers=8, multiprocessing_context="spawn",
                                  )
        valloader = DataLoader(dataset, sampler=val_sampler,shuffle=False,
                                batch_size=1,collate_fn=double_collate,
                                  num_workers=1, multiprocessing_context="spawn",
                                  )
        val_len=len(valloader)-1
    assert len(valloader)>0

    # %%Visualize Input Data
    gpby = dataset.df.group_by(pl.col("Nr"))
    colors = plt.cm.jet(np.linspace(0,1,len(gpby.len())))
    
    for i,group in enumerate(gpby):
        if i==0:
            rtemp = group[1].plot.line(x="time", y="r_temp")
            vtemp = group[1].plot.line(x="time", y="v_temp")
            atemp = group[1].plot.line(x="time", y="TT_TU")
            
        else:
            rtemp *= group[1].plot.line(x="time", y="r_temp")
            atemp *= group[1].plot.line(x="time", y="TT_TU")
            vtemp *= group[1].plot.line(x="time", y="v_temp")
            
        if i>=3:
            break
    
    plots = atemp+rtemp+vtemp
    plots.cols(1)
    hvplot.save(plots, "inputs.html")
    # %%Load and Configure Model
    
    transformer_model = TSTrafo(pred_length=dataset.pred_length,
                    input_size=2,
                    lags_seq=dataset.lag_list,
                    num_time_feat=1+len(dataset.other),
                    num_static_cat_feat=0,
                    cardinality=len(timeseries_idx),
                    embedding_dim=None,
                    enc_layers=6,
                    dec_layers=6,
                    model_dim=64
                    ).cuda()
    
    lstm_model = LSTM(seq_length=dataset.seq_length,
                lag_list=dataset.lag_list,
                pred_length=dataset.pred_length,
                input_size =3+len(dataset.other),
                hidden_size=12,
                num_layers=3).cuda()
    
    fcnn_model = FCNN(dataset.seq_length+max(dataset.lag_list),
                      dataset.pred_length).cuda()
    all_models = [fcnn_model,lstm_model, transformer_model]
    print([count_params(x) for x in all_models])
    loss = MSELoss()
    optimizers = [AdamW(x.parameters(), lr=6e-4,
                         betas=(0.9, 0.95), weight_decay=1e-1)
                         for x in all_models]
    # %% Train Model
    """
    testlosses = train(all_models, optimizers, loss,12,trainloader, testloader,writer,tags=["FCNN","LSTM", "Trafo"])
    print(testlosses)
    outpaths = [os.path.join(os.path.split(path)[0],
                           "../saved_models/{}{}".format(x,timestamp))
                           for x in ["FCNN", "LSTM", "Trafo"]]
    [torch.save(model, outpath) for model,outpath in zip(all_models,outpaths)]
    """
    
    # %% compare models
    intrv = dataset.seq_length+max(dataset.lag_list)
    fig, ax = plt.subplots(3,transformer_model.config.input_size,figsize=(10,10), sharex=True)
    
    for j,saved_name in enumerate([x+timestamp for x in ["FCNN", "LSTM", "Trafo"]]):
        model = torch.load(os.path.join(os.getcwd(),"saved_models",saved_name))
        model.eval()
        model = model.cpu()
        
        try: 
            stack = np.load(os.path.join(os.getcwd(),"results",saved_name+".npz")) 
            stack = [torch.Tensor(stack[key]) for key in ["timestamps","inputs","predictions","target", "zhl_ids"]]
            zhl_ids = np.hstack([[x for _ in range(dataset.pred_length)]
                                    for x in stack[4]])
        except FileNotFoundError:
            results = evaluate(model, valloader)
            stack = [torch.cat([x[0][1] for x in results]),#timestamps
                torch.cat([x[0][3] for x in results]),#inputs
                torch.stack([x[1].squeeze() for x in results]),#predictions
                torch.stack([x[2].squeeze() for x in results]),#targets
                torch.cat([x[0][2] for x in results]),#meter IDs
                ]
            zhl_ids = np.hstack([[x for _ in range(dataset.pred_length)]
                                    for x in stack[4]])
            np.savez_compressed(os.path.join(os.getcwd(),"results",saved_name),
                            inputs = stack[1].numpy(), timestamps = stack[0].numpy(),
                            target = stack[3].numpy(), predictions=stack[2].numpy(),
                            zhl_ids = stack[4].numpy(),)
            
        colors = plt.cm.jet(np.linspace(0,4,len(valloader)))
        if transformer_model.config.input_size==1:
            inputs = torch.stack((stack[0][:,:intrv],
                                stack[1][:,:intrv] ))
            
            predictions = torch.stack((stack[0][:,intrv:],
                                    stack[2]))
            targets = torch.stack((stack[0][:,intrv:],
                                    stack[3]))
            for i in range(val_len)[::int(val_len/3)]:
                assert torch.all(torch.sort(inputs[0,i])[0]==inputs[0,i]),inputs[0,i]
                ax[j].plot(inputs[0,i,-10:], inputs[1,i,-10:],c=colors[i%25],label="past" if j+i==0 else None)
                ax[j].plot(predictions[0,i],predictions[1,i],":",c=colors[i%25],label="predictions" if j+i==0 else None)
                ax[j].plot(predictions[0,i],targets[1,i],"--",c=colors[i%25],label="target" if j+i==0 else None)


        else:
            inputs = torch.stack([stack[0][:,:intrv].squeeze()]+
                                [x[:,:intrv] for x in stack[1].permute(2,0,1) ])
            
            predictions = torch.stack([stack[0][:,intrv:].squeeze()]+
                                    [x for x in stack[2].permute(2,0,1)])
            
           
                

            targets = torch.stack([stack[0][:,intrv:].squeeze()]+
                                    [x for x in stack[3].permute(2,0,1)])
            for i in range(val_len)[::int(val_len/3)]:
                for var in range(1,len(inputs)):
                    #assert torch.all(torch.sort(inputs[0,i])[0]==inputs[0,i]),inputs[0,i]
                    ax[j,var-1].plot(inputs[0,i,-10:], inputs[var,i,-10:],c=colors[i%25],label="past" if j+i==0 else None)
                    ax[j,var-1].plot(predictions[0,i],predictions[var,i],":",c=colors[i%25],label="predictions" if j+i==0 else None)
                    ax[j,var-1].plot(predictions[0,i],targets[var,i],"--",c=colors[i%25],label="target" if j+i==0 else None)
                    ax[j,var-1].set_title(saved_name.replace(timestamp,"")+[" Rücklauftemp", " Wärmelast"][var-1])
        print(saved_name, loss(predictions, targets))
        
        if "Trafo" in saved_name:
            ax[0,1].legend()#just so that it is done only once
            res_df = pl.concat([pl.from_dict(dict(zip(["timestamps","r_temp","Leistung"], x)))
                                                    for x in 
                                        predictions.numpy().transpose(2,0,1)])
            res_df = res_df.with_columns(pl.col("timestamps").cast(int))
            res_df = res_df.with_columns(pl.lit(zhl_ids).alias("Nr"))
            valid_ids = res_df["Nr"].unique().sort()
            res_df = res_df.filter(pl.col("Nr").is_between(valid_ids[0], valid_ids[4]))
            res_df = res_df.join(dataset.df.select(pl.col("timestamps","TT_TU")),on="timestamps")
            #because I can have multiple TS from the same meter
            res_df = res_df.group_by("Nr","timestamps").mean().sort(pl.col("timestamps"))
            #res_df = res_df.with_columns(pl.lit(dataset.start).dt.offset_by(pl.col("timestamps")).alias("time"))
            leistung = res_df.plot.line(x="timestamps", y="Leistung",by="Nr")
            r_temp = res_df.plot.line(x="timestamps", y="r_temp",by="Nr")
            plot = leistung+r_temp
            plot.cols(1)
    
    plt.show()
    fig.savefig("compare_models.png")
    
# %%
