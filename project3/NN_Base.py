def make_model(dataX, lay, active="linear"):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Sequential

    # desc model
    descModel = Sequential()
    descModel.add(Dense(lay[0][0], input_dim=dataX.shape[1], activation=lay[0][1]))
    for l in lay[1:]:
        descModel.add(Dense(l[0], activation=l[1]))
    descModel.add(Dense(dataX.shape[1], activation=active))

    descModel.summary()

    return descModel


def train(model, epo, x, save=False, retrain=False):
    from tensorflow.keras.optimizers import Adam
    import pickle

    if retrain is False:
        model.compile(loss='MeanSquaredError', optimizer="adam")

    history = model.fit(x, x, validation_split=0.1, epochs=epo, batch_size=768)

    if not retrain is False:
        for key in history.history.keys(): retrain[key].extend(history.history[key])
        history.history=retrain

    # save the results
    if save:
        history.model.save('model.h5')
        with open(r"history.pickle", "wb") as file: pickle.dump(history.history, file)

    return model, history.history


def load_model_cos():
    from tensorflow.keras.models import load_model
    import pickle

    model = load_model('model.h5')
    with open(r"history.pickle", "rb") as file: history = pickle.load(file)

    return model, history


def graph_NN(his, leg, save=False,):
    from matplotlib import pyplot as plt

    for i, h in enumerate(his):
        plt.plot(h, linewidth=4-3*i/len(leg))
    plt.legend(leg, loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    # plt.ylim(top=1, bottom=0.9)
    if save: plt.savefig(r"savedPNG/ "+str(leg))
    plt.show()


def initial_load(f_name, lim=None, start=0, load_trans=False):
    from pandas import read_csv, to_numeric
    from numpy import array as np_array
    path=r"input_data/Serialized_pickle/"
    k=range(1, start)
    if start == 0: k=None
    if load_trans:
        read_d=read_csv(path + f_name + ".csv", skiprows=k, nrows=lim)\
            .apply(lambda x: to_numeric(x, errors='coerce')).fillna(0).astype("float32")
        return np_array(read_d.set_index("TransactionID")), read_d[["TransactionID"]]
    return np_array(read_csv(path + f_name + ".csv"
            , skiprows=k, nrows=lim).astype("float32").set_index("TransactionID"))


def load_layers():
    imgLayers = [
        [250, "relu"],
        [125, "relu"],
        [25, "relu"],
        [125, "relu"],
        [250, "relu"],
    ]

    return imgLayers


def find_fraud(auto_model, create_load="create"):
    import pickle
    import numpy as np
    from sklearn.utils import shuffle
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Sequential

    if create_load == "create":
        data_norm = initial_load("normalTrainData", 40000, 200000)
        data_norm = auto_model.predict(data_norm) - data_norm
        for i in range(len(data_norm[0])):
            v = data_norm[:, i]
            data_norm[:, i] = (v - v.min()) / (v.max() - v.min())
        data_fraud = initial_load("fraudTrainData")
        data_fraud = auto_model.predict(data_fraud) - data_fraud
        for i in range(len(data_fraud[0])):
            v = data_fraud[:, i]
            data_fraud[:, i] = (v - v.min()) / (v.max() - v.min())

        data_x = np.concatenate([data_norm,data_fraud])
        data_y = np.concatenate([np.zeros(len(data_norm)),np.ones(len(data_fraud))])
        data_x, data_y = shuffle(data_x, data_y)

        # desc model
        descModel = Sequential()
        descModel.add(Dense(64, input_dim=data_x.shape[1], activation="relu"))
        descModel.add(Dense(32, activation="relu"))
        descModel.add(Dense(16, activation="linear"))
        descModel.add(Dense(2, activation="relu"))
        descModel.add(Dense(1, activation="sigmoid"))

        descModel.compile(loss='binary_crossentropy', optimizer="adam")
        history = descModel.fit(data_x, data_y, validation_split=0.2, epochs=40, batch_size=500)
        history.model.save('fraud_detect_model.h5')
        with open(r"fraud_detect_history.pickle", "wb") as file: pickle.dump(history.history, file)

        graph_NN([history.history['loss'], history.history['val_loss']]
             , ["F_loss", "F_val_loss"], save=False)
        return descModel, history.history

    from tensorflow.keras.models import load_model
    import pickle
    descModel = load_model('fraud_detect_model.h5')
    with open(r"fraud_detect_history.pickle", "rb") as file:
        fraud_history = pickle.load(file)

    graph_NN([fraud_history['loss'], fraud_history['val_loss']]
             , ["F_loss", "F_val_loss"], save=False)
    return descModel, fraud_history


def print_sample_acc(model):
    # calculating mean error
    from numpy import array as np_array
    data_loaded = initial_load("normalTrainData", 50000, 300000)
    predsNorm = model.predict(data_loaded)
    delta=np_array([abs(xp-x) for x, xp in zip(data_loaded, predsNorm)])
    predsNorm = delta.mean()/abs(data_loaded.mean())

    del data_loaded
    data_loaded = initial_load("fraudTrainData")
    predsFraud = model.predict(data_loaded)
    delta=np_array([abs(xp-x) for x, xp in zip(data_loaded, predsFraud)])
    predsFraud = delta.mean()/abs(data_loaded.mean())

    print("normal error should be 0: ", predsNorm*100)
    print("fraud error (shouldN'T be 0): ", predsFraud*100)
    print("difference (should be far from 1): ", predsFraud/predsNorm)


if __name__ == '__main__':
    data_loaded = initial_load("normalTrainData", 100000)
    if 1==1:
        model = make_model(data_loaded, load_layers(), active="sigmoid")
        model, history = train(model, 20, data_loaded, save=True)
    elif 1==1:
        model, history = load_model_cos()
        # model, history = train(model, 30, data_loaded, save=True, retrain=history)
    del data_loaded

    print_sample_acc(model)
    graph_NN([history['val_loss'], history['loss']], ["val_loss", "loss"], save=False)

    # model to detect fraud
    fraud_model, fraud_history = find_fraud(model, create_load="load")

    # outputing test results
    data_test, transID = initial_load("testData", load_trans=True)
    data_test = model.predict(data_test) - data_test
    for i in range(len(data_test[0])):
        v = data_test[:, i]
        data_test[:, i] = (v - v.min()) / (v.max() - v.min())
    isFraud_test = fraud_model.predict(data_test)
    for i, a in enumerate(isFraud_test):
        if a<=0.5: isFraud_test[i]=0
        else: isFraud_test[i]=1
    transID["isFraud"] = isFraud_test
    transID.to_csv("test_results.csv")
    print(isFraud_test.mean())
