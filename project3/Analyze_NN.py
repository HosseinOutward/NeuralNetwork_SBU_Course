from project3.NN_Base import *


def loadRawData(lim=None, start=None):
    from pickle import load as pk_load
    l_dict=pk_load(open("Label_dict.pickle", "rb"))
    def load_csv_file(f_name):
        from pandas import read_csv
        if start==None:
            return read_csv(r"input_data/" + f_name + ".csv", nrows=lim)
        return read_csv(r"input_data/" + f_name + ".csv", skiprows=range(1, start),nrows=lim)

    def clean_cat(data_set):
        cate_k = data_set.select_dtypes('O').astype('category').keys()
        for col in data_set:
            if col in cate_k: data_set[col].replace(l_dict[col], inplace=True)
        return data_set

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

    # ***************
    train, test = clean_cat(train), clean_cat(test)
    fraud_mask = train['isFraud'] == 1
    normal_mask = train['isFraud'] == 0
    train.drop('isFraud', axis=1, inplace=True)
    fraud = train[fraud_mask]
    train = train[normal_mask]

    return train, fraud, test


def create_comp_csv():
    from pandas import DataFrame
    train, farud, _ = loadRawData()

    final_dict={"CSV_index": ["mean1", "mean2", "var1", "var2", "null1",
                "null2", "d-mean", "d-var", "d-null"]}
    for col in train:
        a, b = train[col], farud[col]
        mean1=a.mean()
        var1=a.var()
        null1=1 - a.count()/len(a)

        mean2=b.mean()
        var2=b.var()
        null2=1 - b.count()/len(b)

        final_dict[col]=[mean1, mean2, var1, var2, null1, null2,
                        mean2/mean1-1, (var2-var1)/mean1, (null1-null2)/mean1]

    final_dict=DataFrame.from_dict(final_dict)\
        .set_index("CSV_index").astype("float64").transpose()
    final_dict.to_csv("comparision.csv")


def compare_dataset():
    from pandas import read_csv
    com=read_csv(r"comparision.csv").transpose()
    # i=0;a=0.1;b=0.2
    # for col in com:
    #     d=com[col][["d-mean", "d-var", "d-null"]].values
    #     if (abs(d[0])>a and abs(d[1])>a and abs(d[2])>a) or \
    #             abs(d[0])>b or abs(d[1])>b or abs(d[2])>b:
    i=0;a=0.3;b=0.6
    for col in com:
        d=com[col][["d-mean", "d-var", "d-null"]].values
        if (abs(d[0])>a and abs(d[1])>a and abs(d[2])>a) or \
                abs(d[0])>b or abs(d[1])>b or abs(d[2])>b:
            i+=1
            print(col)
            print(*com[col][["d-mean", "d-var", "d-null"]].values)
            print("****")
    print(i)


if __name__ == '__main__':
    compare_dataset()
