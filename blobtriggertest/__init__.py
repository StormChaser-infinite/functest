import logging
import pandas as pd
import numpy as np
import azure.functions as func
from azure.storage.blob import ContainerClient
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import math



def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes\n"
                 )
    
    df = pd.read_csv(myblob)
    print(df.head())
    output = df.to_csv()

    blobService = ContainerClient(account_url = "https://funcdemo.blob.core.windows.net", 
                                   credential= "e4MRGQUsGoQLwqIQw2pw5fEVSonqVoSpMJV1X0QSZ6gaYmXTaE6aLdz4n6a8BD18wmRa/qbSsU5I+AStB1JRKg==",
                                   container_name = "outputs")

    blobService.upload_blob('OutFilePy.csv', output, overwrite=True, encoding='utf-8')

    # blob_block = ContainerClient.from_connection_string(
    # conn_str= "DefaultEndpointsProtocol=https;AccountName=funcdemo;AccountKey=e4MRGQUsGoQLwqIQw2pw5fEVSonqVoSpMJV1X0QSZ6gaYmXTaE6aLdz4n6a8BD18wmRa/qbSsU5I+AStB1JRKg==;EndpointSuffix=core.windows.net",
    # container_name= 'outputs'
    # )

    # blobService = BlockBlobService(account_name=accountName, account_key=accountKey)
    # # blobService.create_blob_from_text('outputs', 'OutFilePy.csv', output)
    # blob_block.upload_blob('outputs.csv', output, overwrite=True, encoding='utf-8')

    
