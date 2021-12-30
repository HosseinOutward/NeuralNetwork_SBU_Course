# load data
def loadRawData(lim=False):
    from pandas import read_csv
    import os
    import cv2
    from numpy import array

    if lim is False: lim=len(os.listdir(r"input_data/pics"))

    images=[]
    for filename in os.listdir(r"input_data/pics")[:lim]:
        images.append(cv2.resize( cv2.imread(r"input_data/pics/"+filename), (64, 64)))
    images=array(images)

    # image_id, n_city, bed, bath, sqft, price
    desc = read_csv(r"input_data/desc.csv", usecols=["image_id", "n_city", "bed", "bath", "sqft", "price"])[:lim]

    return desc, images


# load and clean CSV data
# image_id, n_city, bed, bath, sqft, price
def cleanDesc(desc):
    from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
    from numpy import hstack, array

    halfbaths=[0 for i in range(len(desc["bath"]))]; baths=[0 for i in range(len(desc["bath"]))]
    for i, bath in enumerate(desc["bath"]):
        halfbaths[i] = int(str(bath).split(".")[1])
        baths[i] = int(bath)
    desc["bath"]=array(baths)
    desc["halfbath"]=array(halfbaths)

    conti=["bed", "bath", "halfbath", "sqft"]
    categ=["n_city"]

    scalerObj = MinMaxScaler()
    continuousData = scalerObj.fit_transform(desc[conti])

    zipBinarizerObj = LabelBinarizer()
    categoricalData = zipBinarizerObj.fit_transform(desc[categ])

    wholeDataScaled = hstack([continuousData, categoricalData])

    pMax = desc["price"].max()
    def priceScale(x): return round(x*pMax, 1)
    prices = desc["price"]/pMax

    image_id = desc["image_id"]

    return image_id, prices, wholeDataScaled, [conti, categ], priceScale, scalerObj, zipBinarizerObj


# load and clean CSV data
# image_id, n_city, bed, bath, sqft, price
def cleanImage(images, bw=True, hEqu=True):
    from numpy import array
    import cv2

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


# take config, load, train and output the model along with history
def make_image_model(imagesX, imgLay, mergeLay, regress=False, active="linear"):
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

    x = Dense(mergeLay[0][0], activation=mergeLay[0][1])(x)

    if regress: x = Dense(1, activation=active)(x)

    imageModel = Model(inputs, x)

    return imageModel


def make_desc_model(dataX, descLay, mergeLay, regress=False, active="linear"):
    from tensorflow.keras.layers import Dense
    from tensorflow.keras import Sequential

    # desc model
    descModel = Sequential()
    descModel.add(Dense(descLay[0][0], input_dim=dataX.shape[1], activation=descLay[0][1]))
    for l in descLay[1:]:
        descModel.add(Dense(l[0], activation=l[1]))

    descModel.add(Dense(mergeLay[0][0], activation=mergeLay[0][1]))

    if regress: descModel.add(Dense(1, activation=active))

    return descModel


def make_whole_model(imagesX, dataX, descLay, imgLay, mergeLay, active="linear"):
    from tensorflow.keras.layers import Dense, concatenate
    from tensorflow.keras import Model

    # combine the 2 nets
    imageModel, descModel = make_image_model(imagesX, imgLay, mergeLay), make_desc_model(dataX, descLay, mergeLay)

    combinedModel=concatenate([imageModel.output, descModel.output])
    x = Dense(mergeLay[1][0], activation=mergeLay[1][1])(combinedModel)
    for l in mergeLay[2:]:
        x = Dense(l[0], activation=l[1])(x)

    x = Dense(1, activation=active)(x)

    model = Model(inputs=[imageModel.input, descModel.input], outputs=x)
    # ***************************

    model.summary()
    return model


def train(model, epo, x, y, save=False, retrain=False):
    from tensorflow.keras.optimizers import Adam
    import pickle

    if retrain is False:
        opt = Adam(lr=1e-3, decay=1e-3 / 200)
        model.compile(loss='mean_absolute_percentage_error', optimizer=opt)

    history = model.fit(x=x, y=y, validation_split=0.2, epochs=epo, batch_size=16)

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
    plt.ylabel('loss_val')
    plt.xlabel('epoch')
    # plt.ylim(top=0.9, bottom=0.8)
    if save: plt.savefig(r"savedPNG/ "+str(leg))
    plt.show()


def load_layers():
    # image layers____
    descLayers = [
        [32, "relu"],
        [16, "relu"],
    ]

    # desc layers____
    imgLayers = [
        [ # conv2d
            [16, "relu"],
            [32, "relu"],
            [64, "relu"],
        ],
        [ # dense
            [32, "relu"],
            [16, "relu"],
        ]
    ]

    # desc layers____
    mergeLayers = [
        [8, "relu"],
        [4, "relu"],
    ]

    return descLayers, imgLayers, mergeLayers


if __name__ == '__main__':
    desc, images = loadRawData()
    image_id, prices, desc, keys, priceScale, scalerObj, zipBinarizerObj = cleanDesc(desc)
    images = cleanImage(images)

    from sklearn.utils import shuffle
    images, desc, prices=shuffle(images, desc, prices)

    descLayers, imgLayers, mergeLayers = load_layers()

    if 0==1:
        model = make_whole_model(images, desc, descLayers, imgLayers, mergeLayers)
        model, history = train(model, 100, [images, desc], prices, save=True)
    elif 1==1:
        model, history = load_model_cos()
        # model, history = train(model, 88, [images, desc], prices, save=True, retrain=history)

    graph_NN([history['val_loss'],history['loss']], ["val_loss", "loss"], save=False)

    test_size=-len(images)//5
    preds = model.predict( [images[test_size:], desc[test_size:]] )
    diff = abs(1 - preds.flatten() / prices[test_size:]) * 100
    avr =diff.mean()

    print("**************")
    print(avr)
