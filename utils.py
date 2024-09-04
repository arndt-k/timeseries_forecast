import torch
import os
import polars as pl
import numpy as np
from datetime import datetime

from torch.utils.data import Dataset,default_collate



    
class FWData(Dataset):
    def __init__(self,path, seq_length, lag_list, pred_length,other=[],limit=None,scale=True):
        """Fernwärme Daten. Map-Style Dataset, loaded completely into memory to extract chunks

        Args:
            path (string): path to data in csv file
            seq_length (int): length of continuous past data sequence
            lag_list (list): lag indices for time-lagged values to be added to inputs
            pred_length (int): length of continuous timeseries to predict
            other (list, optional): column names for ancillary variables to use. Defaults to [].
            limit (int, optional): truncate for quicker runs. Defaults to None.
            scale (bool, optional): if the data should be minmax scaled. Defaults to True
        """
        self.seq_length = seq_length
        self.lag_list = lag_list
        self.pred_length = pred_length
        self.df = pl.read_csv(path,separator=";")
        self.dwd = pl.read_csv(os.path.join(os.path.split(path)[0],
            "../stundenwerte_TU_01303_19510101_20231231_hist/produkt_tu_stunde_19510101_20231231_01303.txt"),
            separator=";")
        #process dwd temperature data
        self.dwd = self.dwd.with_columns(pl.col("MESS_DATUM").cast(str)+"00")
        self.dwd = self.dwd.rename({"MESS_DATUM":"time"})
        self.dwd = self.dwd.with_columns(pl.col("time").str.to_datetime(format="%Y%m%d%H%M",strict=True))
        
        #drop non-informative columns
        self.df = self.df.drop(["Inventarnummer", "Zaehlertyp", "Temperatur 3", "Temperatur 4"])
        self.df = self.df[(self.df.select(pl.all().n_unique())>1).to_numpy().squeeze()]
        #make shorter names
        self.df.columns = ["Nr", "time", "Durchfluss", "En", "En8", "En9",
                        "Leistung", "Masse", "h", "h_err", "v_temp", "r_temp",
                        "vol", "info"]
        #convert to datetime and drop rows with bad formatting or outliers
        self.df = self.df.with_columns(pl.col("time").str.to_datetime(strict=False))
        self.df = self.df.filter(pl.col("time").is_not_null()
                            ).filter(pl.col("time")>datetime(2023,9,1,1,0))
        self.df = self.df.drop_nulls()
        assert len(self.df)>0
        #unclude the air temperature from dwd
        self.df = self.df.join(self.dwd,on="time")
        
        if limit is not None:
            self.df = self.df[:limit]
        
        #so i can return all valid indices
        self.zhl = self.df.select(pl.col("Nr")).unique().to_numpy().squeeze()
        #minmax scaling, saving factors for later rescaling
        self.mins = self.df.select([ "Durchfluss", "En" , "En9",
                        "Leistung", "Masse", "h", "h_err", "v_temp", "r_temp",
                        "vol","TT_TU"]).min()
        self.maxs = self.df.select(["Durchfluss", "En",  "En9",
                        "Leistung", "Masse", "h", "h_err", "v_temp", "r_temp",
                        "vol","TT_TU"]).max()
        if scale:
            self.df = self.df.with_columns([(pl.col(x)-self.mins.select(pl.col(x)))/
                                        (self.maxs-self.mins).select(pl.col(x)) 
                                        for x in self.maxs.columns])
        self.start = self.df["time"].min()
        self.df = self.df.with_columns(((pl.col("time") - self.start)/1e6/3600
                                        ).cast(pl.Int64).alias("timestamps"))
        self.other=other
        """
        un_ids = self.df.select(pl.col("Nr")).unique().to_numpy()
        zhl_dict = dict(zip(un_ids, range(len(un_ids))))
        self.df = self.df.with_columns(pl.col("Nr"))
        """
    def __len__(self):
        return len(self.df)
    
    def rescale(self):
        return self.df.with_columns([(pl.col(x)*(self.maxs-self.mins).select(pl.col(x))+
                                      self.mins.select(pl.col(x))) 
                                        for x in self.maxs.columns])
    
    def __getitem__(self, idx):
        #which timesieres and where in timeseries
        ts_id, zhl_id = idx
        #get only relevant timeseries
        now_df = self.df.filter(pl.col("Nr")==zhl_id)
        if ts_id-self.seq_length-max(self.lag_list)<0:
            raise ValueError("Trying to access point before start of timeseries."+
                              "ts_id {} too small for seq length {} and max lag {}".format(
                                  ts_id, self.seq_length, max(self.lag_list)))
        
        #create indices to slice timeseries
        seq =np.arange(ts_id-max(self.lag_list)-self.seq_length,
                       ts_id+self.pred_length)
        
        if max(seq)>=len(now_df):
            #not sure how to prevent this for variable length series
            return
        #the chunk of the timeseries
        now_df = now_df[seq]
        #time since the earliest sample in the ds in hours
        timestamps = now_df.select(pl.col("timestamps")).to_numpy(writable=True)
        other = now_df.select(pl.col(self.other)).to_numpy(writable=True)
        assert len(timestamps.shape)==len(other.shape)
        
        return (other, #ancillary features
                timestamps, #timestamps
                zhl_id, #meter ids
                now_df[["r_temp","Leistung"]][:-self.pred_length].to_numpy(writable=True), #values
                now_df[["r_temp","Leistung"]][-self.pred_length:].to_numpy(writable=True) #prediction target
        )
        

def double_collate(x):
    """Removes nonexistent samples and sets every dtype to double

    Args:
        x (tuple): output of dataset.__getitem__

    Returns:
        list: the batches usable by the model
    """
    x = [y for y in x if y is not None]
    if len(x)==0:
        return
    return [y.type(torch.float32) for y in default_collate(x)]

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)