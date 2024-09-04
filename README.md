# District Heating Forecast  
Small case study to asses feasibility of different neural networks to forecast return temperature in district heating networks. Three different NNs are compared:  
- Fully Connected NN
- LSTM
- Transformer

All use past return temperature as input sequence, with some optional additional future sequences for the transformer, in this case outside temperature and input temperature.
The dataset is fully loaded into memory as a `polars` dataframe and then accessed in batches using a `pytorch` dataloader.
Using polars requires the dataloader to use the "spawn" method for multiprocessing, which seems to limit the maximum possible size of the LSTM, such that it can never achieve good results.  

The following plot shows sequences of three random meters for the outside temperature, the input temperature and the return temperature.  
![alt text](https://github.com/arndt-k/timeseries_forecast/blob/main/inputs.png?raw=true)  
The following plot compares the different models, showing that the Transformer can produce the historical timeseries reasonably correctly, that the FCNN is surprisingly good considering its simplicity, and that the LSTM can not be fairly compared, as mentioned above.  
![alt text](https://github.com/arndt-k/timeseries_forecast/blob/main/compare_models.png?raw=true)
