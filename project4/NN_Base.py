def loadRawInfo(path, lim=None):
    from os import listdir
    from pandas import read_csv

    path = r"input_data/"+path+"/"
    k = 0
    if "M"!=listdir(path)[0][-6] and "F"!=listdir(path)[0][-6]: k=1; csv_f=read_csv(r"input_data/result.csv")
    data_list = []

    for i, file_name in enumerate(listdir(path)):
        if i == lim: break
        data_list.append([file_name, file_name[:-6+k], {"M":0, "F":1}[file_name[-6+k]],])
        if k==1: data_list[-1].append({"A":0, "H":1, "S":2, "W":3, "N":4}\
                        [csv_f.loc[csv_f['Id'] == data_list[-1][0][:-4]].values[0][1]])
        elif k==0: data_list[-1].append({"A":0, "H":1, "S":2, "W":3, "N":4}[file_name[-5]])

    # file_adr, File_ID, Gender, Emotions
    return (data_list, path)


def clean_data(info_list, path):
    from sklearn.utils import shuffle
    from numpy import array as np_array
    from random import random

    def get_comp_data(f_name, func, step, max_len=1000, sr_divide=3):
        import librosa as lrs
        from numpy import pad as np_pad
        max_len = max_len // sr_divide
        wave, sr = lrs.load(path+f_name, mono=True, sr=None)
        wave=wave[step::sr_divide]
        data_comp = eval(["lrs.feature.mfcc(wave, sr=sr)",
                         "lrs.feature.chroma_stft(wave, sr=sr)",
                         "lrs.feature.zero_crossing_rate(wave)",][func])

        # pad the remaining ones or drop rest
        if max_len <= data_comp.shape[1]:
            data_comp = data_comp[:, :max_len]
        else:
            pad_width = max_len - data_comp.shape[1]
            data_comp = np_pad(data_comp, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return data_comp

    x_cleaned = [[],[],[],[]]
    y_cleaned = []
    id_data = []
    #{"A": 0, "H": 1, "S": 2, "W": 3, "N": 4}
    multiplier={0:1, 1:3, 2:2, 3:3, 4:1}
    # multiplier={0:1, 1:1, 2:1, 3:1, 4:1}
    if "Test" in path: multiplier={0:1.2, 1:4, 2:3, 3:6, 4:1}
    for sample in info_list:
        for step in [0,1,2]:
            temp_x=[]
            temp_x.append(get_comp_data(sample[0], 0, step).transpose())
            temp_x.append(get_comp_data(sample[0], 1, step).transpose())
            temp_x.append(get_comp_data(sample[0], 2, step).transpose())
            temp_x.append(np_array([sample[2]]))
            temp_y = [0] * 5
            temp_y[sample[3]] = 1
            temp_id = sample[1]
            loop_count=multiplier[sample[3]]
            if random() < loop_count-round(loop_count): loop_count+=1
            loop_count=round(loop_count)
            for i in range(loop_count):
                id_data.append(temp_id)
                y_cleaned.append(temp_y)
                x_cleaned[0].append(temp_x[0])
                x_cleaned[1].append(temp_x[1])
                x_cleaned[2].append(temp_x[2])
                x_cleaned[3].append(temp_x[3])

    for i in range(len(x_cleaned)-1):
        x_cleaned[i]=np_array(x_cleaned[i])
        for j in range(len(x_cleaned[i][0])):
            for k in range(len(x_cleaned[i][0][j])):
                v = x_cleaned[i][:,j,k]
                x_cleaned[i][:,j,k] = (v - v.min()) / (v.max() - v.min())
    x_cleaned[3]=np_array(x_cleaned[3])

    x_cleaned[0], x_cleaned[1], x_cleaned[2], x_cleaned[3], y_cleaned, id_data=\
        shuffle(x_cleaned[0], x_cleaned[1], x_cleaned[2], x_cleaned[3], y_cleaned, id_data)

    return x_cleaned, y_cleaned, id_data


def make_model(data_sets, output_data, lay, active="linear", lmc=3):
    from keras.layers import Conv2D, LSTM, Dense,\
        concatenate, BatchNormalization, MaxPooling2D, Flatten
    from keras import Sequential, Model

    model_list=[]
    for i, data in enumerate(data_sets[:lmc]):
        model = Sequential()
        model.add(LSTM(lay[i][0][0][0], activation=lay[i][0][0][1], return_sequences=True, input_shape=data[0].shape))
        for l in lay[i][0][1:]:
            model.add(LSTM(l[0], activation=l[1]))
        for l in lay[i][1][1:]:
            model.add(Dense(l[0], activation=l[1]))

        model_list.append(model)

    for i, data in enumerate(data_sets[lmc:lmc+1], start=lmc):
        model = Sequential()
        model.add(Dense(lay[i][0][0], input_dim=data.shape[1], activation=lay[i][0][1]))
        for l in lay[i][1:]:
            model.add(Dense(l[0], activation=l[1]))

        model_list.append(model)

    for i in range(2):
        model = Sequential()
        model.add(Conv2D(8, (11, 3), padding="same", activation="relu", input_shape=data_sets[i-2][0].shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(6, 2)))
        model.add(Conv2D(16, (3, 2), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(4, 2)))
        model.add(Conv2D(32, (2, 2), padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model_list.append(model)

    for model in model_list:
        model.add(Dense(lay[-1][0][0], activation=lay[-1][0][1]))

    model = concatenate([a.output for a in model_list])
    for l in lay[-1][1:]:
        model = Dense(l[0], activation=l[1])(model)
    model = Dense(output_data.shape[1], activation=active)(model)

    model = Model(inputs=[a.input for a in model_list], outputs=model)
    model.summary()

    return model


def train(model, epo, data_set, y, valid_x, valid_y, save=False, retrain=False):
    import pickle

    if retrain is False:
        model.compile(loss='categorical_crossentropy', optimizer="adam")

    history = model.fit(data_set, y, validation_data=(valid_x, valid_y), epochs=epo, batch_size=600)

    if not retrain is False:
        for key in history.history.keys(): retrain[key].extend(history.history[key])
        history.history=retrain

    # save the results
    if save:
        history.model.save('model.h5')
        with open(r"history.pickle", "wb") as file: pickle.dump(history.history, file)

    return model, history.history


def print_sample_acc(sample, real_v, ids, model_t):
    from sklearn.metrics import confusion_matrix
    from seaborn import heatmap
    import matplotlib.pyplot as plt

    labels={0: 'Angry', 4: 'Neutr', 1: 'Happy', 2: 'Sad', 3: 'Wonde'}
    predict_v = model_t.predict(sample)

    id_dict={}
    for i, id in enumerate(ids):
        if not id in id_dict.keys(): id_dict[id]=[]
        id_dict[id].append(i)

    a=0; f_pred_v=[]; f_real_v=[]
    for indexes in id_dict.values():
        value_list=[]
        for i in indexes:
            flag=True
            for v2 in value_list:
                if (predict_v[i]==v2).all(): flag=False; break
            if flag :value_list.append(predict_v[i])

        aver_pred=[0]*5
        for v in value_list: aver_pred+=v
        aver_pred/=len(value_list)
        f_real_v.append(real_v[i])
        f_pred_v.append(aver_pred)

    conf_real = []
    conf_pred = []
    for i in range(len(f_real_v)):
        conf_real.append(labels[f_real_v[i].argmax()])
        conf_pred.append(labels[f_pred_v[i].argmax()])

    labels = ["Angry", "Neutr", "Happy", 'Sad', 'Wonde']

    conf_matrix = confusion_matrix(conf_real, conf_pred, labels=labels)

    heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt="d", cmap="Oranges")
    plt.title("confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def load_or_clean_data(size=None, raw=True):
    from numpy import save, load
    path = r'input_data/cleaned_data/'

    if raw:
        train_x, train_y, train_ID = clean_data(*loadRawInfo("Train_data", size))
        test_x, test_y, test_ID = clean_data(*loadRawInfo("Test_data", size))
        for i in range(len(train_x)):
            save(path + 'train_x' + str(i) + '.npy', train_x[i])
            save(path + 'test_x' + str(i) + '.npy', test_x[i])
        save(path + 'train_y.npy', train_y)
        save(path + 'train_ID.npy', train_ID)
        save(path + 'test_y.npy', test_y)
        save(path + 'test_ID.npy', test_ID)

    train_x, test_x = [], []
    for i in range(4):
        train_x.append(load(path + 'train_x' + str(i) + '.npy'))
        test_x.append(load(path + 'test_x' + str(i) + '.npy'))
    train_y = load(path + 'train_y.npy')
    train_ID = load(path + 'train_ID.npy')
    test_y = load(path + 'test_y.npy')
    test_ID = load(path + 'test_ID.npy')

    return train_x, train_y, train_ID, test_x, test_y, test_ID


def load_model_cos():
    from tensorflow.keras.models import load_model
    import pickle

    model = load_model('model.h5')
    with open(r"history.pickle", "rb") as file: history = pickle.load(file)

    def reset_weights(model):
        import tensorflow as tf
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                reset_weights(layer)
                continue
            for k, initializer in layer.__dict__.items():
                if "initializer" not in k:
                    continue
                # find the corresponding variable
                var = getattr(layer, k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
    # reset_weights(model)

    return model, history


def graph_NN(his, leg, save=False,):
    from matplotlib import pyplot as plt

    for i, h in enumerate(his):
        plt.plot(h, linewidth=4-3*i/len(leg))
    plt.legend(leg, loc='upper left')
    plt.title('model accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    if save: plt.savefig(r"savedPNG/ "+str(leg))
    plt.show()


def load_layers():
    layers = [
    # M1
    [
        # lstm
        [
            [64, "tanh"],
            [32, "tanh"],
        # dense
        ], [
            [16, "relu"],
            [8, "relu"],
        ]
    # M2
    ], [
        # lstm
        [
            [32, "tanh"],
            [16, "tanh"],
        # dense
        ], [
            [8, "relu"],
            [4, "relu"],
        ]
    # M3
    ],[
        # lstm
        [
            [16, "tanh"],
            [8, "tanh"],
        # dense
        ], [
            [4, "relu"],
        ]
    # M4
    ],[
        [1, "linear"],
    # merging layer
    ],[
        [64, "relu"],
        [16, "relu"],
        [32, "softmax"],
        [16, "softmax"],
    ]
    ]

    return layers


if __name__ == '__main__':
    # import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    train_x, train_y, train_ID, test_x, test_y, test_ID = load_or_clean_data(raw=False)

    if 0 == 1:
        model = make_model(train_x, train_y, load_layers(), active="sigmoid", lmc=3)
        model, history = train(model, 1, train_x, train_y, test_x, test_y, save=True)
    elif 1 == 1:
        model, history = load_model_cos()
        model.summary()
        model, history = train(model, 10, train_x, train_y, test_x, test_y, save=True, retrain=history)

    graph_NN([history['val_loss'], history['loss']], ["val_loss", "loss"])
    print_sample_acc(test_x, test_y, test_ID, model)
    print_sample_acc(train_x, train_y, train_ID, model)
