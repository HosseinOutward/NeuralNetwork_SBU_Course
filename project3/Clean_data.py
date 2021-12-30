# load data
def loadRawData(lim=None, start=None):
    def load_csv_file(f_name):
        from pandas import read_csv
        if start==None:
            return read_csv(r"input_data/" + f_name + ".csv", nrows=lim)
        return read_csv(r"input_data/" + f_name + ".csv", skiprows=range(1, start),nrows=lim)

    train_ident = load_csv_file("train_identity")
    train_trans = load_csv_file("train_transaction")

    test_ident = load_csv_file("test_identity")
    test_trans = load_csv_file("test_transaction")

    sample_sub = load_csv_file("sample_submission")[["TransactionID"]]

    train = train_trans.join(train_ident.set_index('TransactionID'), on="TransactionID", how='left').set_index('TransactionID')
    test = test_trans.join(test_ident.set_index('TransactionID'), on="TransactionID", how='outer')
    if len(test)!=0:
        test = test.join(sample_sub.set_index('TransactionID'), on="TransactionID", how='right').set_index('TransactionID')

    for col in test:
        test.rename(columns={col:col.replace("-", "_")}, inplace=True)
    for col in train:
        test.rename(columns={col:col.replace("-", "_")}, inplace=True)

    cate_k = train.select_dtypes('O').astype('category').keys()
    for col in train:
        if col in cate_k: continue
        train[col]=train[col].astype("float32")
    cate_k = test.select_dtypes('O').astype('category').keys()
    for col in test:
        if col in cate_k: continue
        test[col]=test[col].astype("float32")

    return train, test[train.keys().drop("isFraud")]


# set a label dictionary for later use
def find_label_dict(size, raw=False):
    from pandas import isnull as isNan, read_csv
    from pickle import dump as pk_dump, load as pk_load

    if not raw: return pk_load(open("Label_dict.pickle", "rb"))

    def load_csv_file(f_name, usecols):
        return read_csv(r"input_data/" + f_name + ".csv",usecols=usecols)

    l_dict={}
    sample=loadRawData(size, None)[0]
    colls = sample.select_dtypes('O').astype('category').keys()

    for col in sample.keys():
        try:
            train_col = load_csv_file("train_transaction", usecols=["TransactionID",col]).\
                join(load_csv_file("train_identity", ["TransactionID"]).\
                set_index('TransactionID'), on="TransactionID", how='left')[col]
        except:
            train_col = load_csv_file("train_transaction", ["TransactionID"]).\
                join(load_csv_file("train_identity", usecols=["TransactionID",col]).\
                set_index('TransactionID'), on="TransactionID", how='left')[col]

        l_dict[col] = {}
        if col in colls:
            for v in train_col.values:
                if (not isNan(v)) and (not v in l_dict[col].keys()):
                    l_dict[col][v]=len(l_dict[col])+1

            dict_min= min(l_dict[col].items(), key=lambda x: x[1])[1]
            dict_max = max(l_dict[col].items(), key=lambda x: x[1])[1]
            for k in l_dict[col]: l_dict[col][k] = (l_dict[col][k]-dict_min)/(dict_max-dict_min)
        else:
            l_dict[col]["max"]=max(train_col)
            l_dict[col]["min"]=min(train_col)

        print(col, "added to label_dict")

    print("label_dict loaded")

    pk_dump(l_dict, open("Label_dict.pickle", "wb"))

    return l_dict


# load and clean CSV data
def clean_data(dataTrain, dataTest, l_dict):
    import pandas as pd

    # ***************
    fraud_mask = dataTrain['isFraud'] == 1
    normal_mask = dataTrain['isFraud'] == 0
    dataTrain.drop('isFraud', axis=1, inplace=True)
    dataTrain_normal = dataTrain[normal_mask]
    dataTrain_fraud = dataTrain[fraud_mask]

    def clean_one_set(data_set):
        cate_k = data_set.select_dtypes('O').astype('category').keys()
        for col in data_set:
            if col in cate_k:
                data_set[col].replace(l_dict[col], inplace=True)
            else:
                s = l_dict[col]
                data_set[col]=data_set[col].sub(s["min"]).div((s["max"] - s["min"]))

            add_c_array=[]
            for a in data_set[col].values:
                if pd.isnull(a): add_c_array.append(1)
                else: add_c_array.append(0)
            data_set[col+"_NAN"] = add_c_array

        data_set = data_set.fillna(0)

        return data_set

    fraudT_Data = clean_one_set(dataTrain_fraud)
    test_Data = clean_one_set(dataTest)
    normT_Data = clean_one_set(dataTrain_normal)

    return normT_Data, fraudT_Data, test_Data


# final function to clean and dump data
def conv_to_clean_file(size=1000):
    from sklearn.utils import shuffle
    from numpy import array as np_array

    path = r"input_data/Serialized_pickle/"
    label_dict=find_label_dict(20000)

    rawTrain, rawTest = loadRawData(size, None)
    normalTrainData, fraudTrainData, testData = clean_data(rawTrain, rawTest, label_dict)
    normalTrainData.to_csv(path + "normalTrainData.csv")
    fraudTrainData.to_csv(path + "fraudTrainData.csv")
    testData.to_csv(path + "testData.csv")

    max_point = 600000 // size
    for i in range(1, max_point):
        rawTrain, rawTest = loadRawData(size, start=i * size)
        if len(rawTest) == 0 and len(rawTrain) == 0: print("break on", i*size);break
        normalTrainData, fraudTrainData, testData = clean_data(rawTrain, rawTest, label_dict)

        normalTrainData = shuffle(normalTrainData)
        fraudTrainData = shuffle(fraudTrainData)
        testData = shuffle(testData)

        normalTrainData.to_csv(path + "normalTrainData.csv", mode='a', header=None)
        fraudTrainData.to_csv(path + "fraudTrainData.csv", mode='a', header=None)
        testData.to_csv(path + "testData.csv", mode='a', header=None)

        print(i*size,"done")

    return np_array(normalTrainData), np_array(fraudTrainData), np_array(testData)


conv_to_clean_file(20000)