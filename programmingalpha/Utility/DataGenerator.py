
def batchGenerator(data:list,batch_size):
    n=len(data)
    ptr=0
    while ptr+batch_size<=n:
        batch=data[ptr:ptr+batch_size]
        yield batch
        ptr+=batch_size
    if ptr<batch_size:
        yield data[ptr:]


