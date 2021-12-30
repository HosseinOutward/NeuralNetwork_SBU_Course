from project2_2.NN_Base import *


def test_mean_results(f, dataX, dataY, dL, mL, r=15, epo=15):
    testInt = -1000
    averages=[]
    hiss=[0 for i in range(epo)]
    for i in range(r):
        model = f(dataX, dL, mL, regress=True)
        _, history = train(model, epo, dataX, dataY, save=False)

        preds = model.predict(dataX[testInt:])
        diff = abs(1 - preds.flatten() / dataY[testInt:]) * 100
        aver = diff.mean()
        averages.append(aver)

        hiss = [a + b for a, b in zip(history["val_loss"], hiss)]

    hiss = [h / r for h in hiss]
    averages = sorted(averages)[:5]

    print(averages)
    graph_NN([hiss], ["model"], save=False)


if __name__ == '__main__':
    desc, images = loadRawData(6000)
    image_id, prices, desc, keys, priceScale, scalerObj, zipBinarizerObj = cleanDesc(desc)
    images = cleanImage(images, bw=True, hEqu=False)

    from sklearn.utils import shuffle
    images, desc, prices=shuffle(images, desc, prices)

    descLayers, imgLayers, mergeLayers = load_layers()

    if 0==1:
        test_mean_results(make_image_model, images, prices, imgLayers, mergeLayers)
    elif 1==1:
        test_mean_results(make_desc_model, desc, prices, descLayers, mergeLayers)
