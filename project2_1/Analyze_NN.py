from project2_1.NN_Base import *


def test_mean_results(f, dataX, dataY, validX, validY, dL, r=3, epo=3):
    testInt = -1000
    averages=[]
    hiss=[0 for i in range(epo)]
    for i in range(r):
        print("loop "+str(i))
        model = f(dataX, dataY, dL)
        _, history = train(model, epo, dataX, dataY, validX, validY, save=True)

        averages.append(max(history["val_accuracy"][-5:]))
        hiss = [a + b for a, b in zip(history["val_accuracy"], hiss)]

    hiss = [h / r for h in hiss]
    averages = sorted(averages)[:5]

    print(averages)
    graph_NN([hiss], ["model"], save=False)

    return averages, hiss


def conv_layer_shape(model, data):
    import matplotlib.pyplot as plt
    import numpy as np

    data_pred = model.predict(data)
    layer_names = [layer.name for layer in model.layers]

    first_layer_activation = data_pred[0]
    # plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, data_pred):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


if __name__ == '__main__':
    map_dict, dataTrain, dataValid, dataTest = loadRawData()

    imgTrain, imgValid, imgTest = cleanImage(dataTrain, dataValid, dataTest, hEqu=False)
    labelTrain, labelValid, labelTest, zipBin = cleanLabels(dataTrain, dataValid, dataTest)

    # from sklearn.utils import shuffle
    # imgTrain, labelTrain = shuffle(imgTrain, labelTrain)
    # imgValid, labelValid = shuffle(imgValid, labelValid)
    # imgTest, labelTest = shuffle(imgTest, labelTest)
    #
    # imgLayers = load_layers()

    # averages, hiss = test_mean_results(make_image_model, imgTrain, labelTrain, imgValid, labelValid, imgLayers)
    # graph_NN([hiss], ["Model"], save=False)
    # print(averages)

    data_x=[]
    for img, lbl in zip(imgTest, labelTest):
        if len(data_x)==43: break
        if lbl.argmax()==len(data_x): data_x.append(img)
    import numpy as np
    data_x=np.array(data_x)

    model, _ = load_model_cos()
    conv_layer_shape(model, data_x)

