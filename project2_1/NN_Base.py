# load data
def loadRawData():
    from pandas import read_csv
    import pickle

    # coords, labels, features, sizes
    dataTrain = pickle.load(open("input_data/train.pickle", "rb"))
    dataValid = pickle.load(open("input_data/valid.pickle", "rb"))
    dataTest = pickle.load(open("input_data/test.pickle", "rb"))

    # ClassId, SignName
    map_dict = read_csv(r"input_data/label_names.csv")

    return map_dict, dataTrain, dataValid, dataTest


# load and clean CSV data
# image_id, n_city, bed, bath, sqft, price
def cleanImage(dataTrain, dataValid, dataTest, bw=True, hEqu=True):
    from numpy import array
    import cv2

    def change_set(images):
        if bw:
            newImg=[]
            for img in images: newImg.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            images = array(newImg)

        if hEqu and bw:
            newImg = []
            for img in images: newImg.append(cv2.equalizeHist(img))
            images = array(newImg)

        images = images / 255.0

        return images

    dTr = change_set(dataTrain["features"])
    dVa = change_set(dataValid["features"])
    dTe = change_set(dataTest["features"])

    return dTr, dVa, dTe


# load and clean CSV data
# image_id, n_city, bed, bath, sqft, price
def cleanLabels(dataTrain, dataValid, dataTest):
    from sklearn.preprocessing import LabelBinarizer
    from numpy import hstack

    hstack((dataTrain["labels"], dataValid["labels"], dataTest["labels"]))

    zipBinarizerObj = LabelBinarizer()
    zipBinarizerObj.fit( hstack((dataTrain["labels"], dataValid["labels"], dataTest["labels"])) )

    labelTrain = zipBinarizerObj.transform(dataTrain["labels"])
    labelValid = zipBinarizerObj.transform(dataValid["labels"])
    labelTest = zipBinarizerObj.transform(dataTest["labels"])

    return labelTrain, labelValid, labelTest, zipBinarizerObj


# take config, load, train and output the model along with history
def make_image_model(imagesX, imageY, imgLay, active="softmax"):
    from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
    from tensorflow.keras import Model

    # image model
    if len(imagesX.shape) == 4: inputs = Input(imagesX.shape[1:])
    else: inputs = Input((imagesX.shape[1], imagesX.shape[2], 1))

    chanDim = -1
    x = inputs
    for l in imgLay[0]:
        x = Conv2D(l[0], (3, 3), padding="same", activation=l[1])(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    for l in imgLay[1][:-1]:
        x = Dense(l[0], activation=l[1])(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Dropout(0.5)(x)

    x = Dense(imageY.shape[1], activation=active)(x)

    imageModel = Model(inputs, x)

    return imageModel


def train(model, epo, x, y, vx, vy, save=False, retrain=False):
    from tensorflow.keras.optimizers import Adam
    import pickle

    if retrain is False:
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(x, y, validation_data=(vx,vy), epochs=epo, batch_size=16)

    if not(retrain is False):
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
    plt.ylim(top=1, bottom=0.9)
    if save: plt.savefig(r"savedPNG/ "+str(leg))
    plt.show()


def load_layers():
    imgLayers = [
        [ # conv2d
            [8, "relu"],
            [16, "relu"],
            [32, "relu"],
        ],
        [ # dense
            [32, "relu"],
            [16, "relu"],
        ]
    ]

    return imgLayers


if __name__ == '__main__':
    map_dict, dataTrain, dataValid, dataTest = loadRawData()

    imgTrain, imgValid, imgTest = cleanImage(dataTrain, dataValid, dataTest)
    labelTrain, labelValid, labelTest, zipBin = cleanLabels(dataTrain, dataValid, dataTest)

    from sklearn.utils import shuffle
    imgTrain, labelTrain = shuffle(imgTrain, labelTrain)
    imgValid, labelValid = shuffle(imgValid, labelValid)
    imgTest, labelTest = shuffle(imgTest, labelTest)

    imgLayers = load_layers()

    if 0==1:
        model = make_image_model(imgTrain, labelTrain, imgLayers)
        model, history = train(model, 150, imgTrain, labelTrain, imgValid, labelValid, save=True)
    elif 1==1:
        model, history = load_model_cos()
        # model, history = train(model, 150, imgTrain, labelTrain, imgValid, labelValid, save=True, retrain=history)

    preds = model.predict( imgTest )
    diff = 1-abs(labelTest-preds)
    avr_diff = diff.mean()

    graph_NN([history['val_accuracy'], history['accuracy'], [avr_diff for i in range(len(history['accuracy']))] ], ["val_accuracy", "accuracy", "test_set"], save=False)
