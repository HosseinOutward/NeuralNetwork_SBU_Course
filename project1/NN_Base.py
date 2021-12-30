# load and clean CSV data
def loadCSV(scale=True):
    from pandas import read_csv
    import numpy as np
    from sklearn import preprocessing

    # load relevant column
    dataset = read_csv("data.csv", usecols=["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                                            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"])

    # map categorical data to numbers
    def mapNames(data, colN, mapp):
        mapp[colN]={}
        for st in data[colN]:
            if not(st in mapp[colN].keys()):
                mapp[colN][st] = len(mapp[colN])+1
    mapping={}
    for colName in ["Geography", "Gender"]:
        mapNames(dataset, colName, mapping)
        dataset = dataset.replace({colName: mapping[colName]})

    # shuffle, scale, split (in-out) the data
    dataKey=dataset.keys()
    dataset=dataset.values
    np.random.shuffle(dataset)

    inX = dataset[:len(dataset), :len(dataset[0])-1]
    if scale:
        inX=preprocessing.scale(inX)
        # maxs=[sum([row[i] for row in inX])/len(inX[0]) for i in range(len(inX[0]))]
        # inX = [[j/maxs[k] for k, j in enumerate(i)] for i in inX]
        # inX = np.array(inX)
    outY = dataset[:len(dataset), len(dataset[0])-1]

    return inX, outY, dataKey


# take config, load, train and output the model along with history
def train(ep, inX, outY, lay, batch=20, opt="RMSprop", actF='sigmoid', test=False, model=True):
    from tensorflow import keras

    if test:
        import numpy as np; import tensorflow as tf; import random as python_random;import os
        os.environ['PYTHONHASHSEED']=str(1234);np.random.seed(1234);python_random.seed(1234);tf.random.set_seed(1234);

    # if a model is not inputted, creat a new one with the config given
    if model is True:
        model = keras.Sequential()

        model.add(keras.layers.Dense(lay[0][0], input_dim=10, activation=lay[0][1]))
        for l in lay[1:]:
            model.add(keras.layers.Dense(l[0], activation=l[1]))
        model.add(keras.layers.Dense(1, activation=actF))

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    history = model.fit(inX, outY, validation_split=0.2, epochs=ep, batch_size=batch)

    return history


# graph the resault(s)
def graph_NN(his, leg, save=False,):
    from matplotlib import pyplot as plt

    for i, h in enumerate(his):
        plt.plot(h, linewidth=4-3*i/len(leg))
    plt.legend(leg, loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim(top=0.9, bottom=0.8)
    if save: plt.savefig(r"savedPNG/ "+str(leg))
    plt.show()


# main function which will find a good seed and train it
# or train an existing model
def run(x, y, lay, bat, opt, act, epo, load=True, set_seed=False, re_train=False):
    if not(load):
        # check different seeds for a good seed
        acc=0
        target=0.8
        while acc<=target+0.06:
            hiss = train(5, x, y, lay, batch=64, opt=opt, actF=act, test=set_seed)
            if hiss.history['val_accuracy'][-1]>target+0.03:
                acc = train(4, x, y, lay, batch=32, opt=opt, actF=act, test=set_seed, model=hiss.model)
                hiss.history['val_accuracy'].extend(acc.history['val_accuracy'])
            if hiss.history['val_accuracy'][-1]>target+0.04:
                acc = train(5, x, y, lay, batch=16, opt=opt, actF=act, test=set_seed, model=acc.model)
                hiss.history['val_accuracy'].extend(acc.history['val_accuracy'])
            if hiss.history['val_accuracy'][-1]>target+0.05:
                acc = train(6, x, y, lay, batch=8, opt=opt, actF=act, test=set_seed, model=acc.model)
                hiss.history['val_accuracy'].extend(acc.history['val_accuracy'])
            if hiss.history['val_accuracy'][-1]>target+0.06:
                acc = train(5, x, y, lay, batch=4, opt=opt, actF=act, test=set_seed, model=acc.model)
                hiss.history['val_accuracy'].extend(acc.history['val_accuracy'])
                acc.history['val_accuracy']=hiss.history['val_accuracy']
                hiss=acc
            acc=hiss.history['val_accuracy'][-1]

        # train the model
        acc=train(epo, x, y, lay, batch=bat, opt=opt, actF=act, test=set_seed, model=hiss.model)
        hiss.history['val_accuracy'].extend(acc.history['val_accuracy'])
        acc.history['val_accuracy']=hiss.history['val_accuracy']
        hiss=acc

        # save the resaults
        model=hiss.model
        model.save('model.h5')
        hiss=hiss.history
        with open(r"history.pickle", "wb") as file: pickle.dump(hiss, file)
        with open(r"xy.pickle", "wb") as file: pickle.dump({"x":x,"y":y}, file)
    else:
        # load results
        from tensorflow import keras
        model = keras.models.load_model('model.h5')
        with open(r"history.pickle", "rb") as file: hiss = pickle.load(file)
        with open(r"xy.pickle", "rb") as file: xy=pickle.load(file)
        x=xy["x"]; y=xy["y"]

    if re_train:
        train(epo, x, y, lay, batch=bat, opt=opt, actF=act, test=set_seed, model=model)
    # graph the new training history
    graph_NN([hiss['val_accuracy']], [[bat, opt, act, epo]], save=True)

    return model, hiss, x, y


if __name__ == '__main__':
    import pickle

    x,y, key=loadCSV()
    layers=[
        [20, None],
        [30, None],
        [20, 'tanh'],
        [20, 'relu'],
        [10, 'relu'],
        [20, 'relu'],
        [10, 'relu'],
    ]

    model, history, x, y = run(x, y, layers, 8, 'adamax', 'softplus', 50, load=True, re_train=False, set_seed=False)
