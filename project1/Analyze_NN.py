from NN_Base import graph_NN, loadCSV, train


# test the effectiveness of a config for r times and averaging it and graph it
# or load result of the last test and show them on graph
def test(x,y,lays, bats, opts_acts, epo, r=1, load=True):
    hist=[]
    legend=[]
    target=0.8
    if not(load):
        for batchSize in bats:
            for optimizer, activeFunc in opts_acts:
                hiss=[0 for i in range(epo)]
                for i in range(1, r+1):
                    trained=train(epo, x, y, lays, batch=batchSize, opt=optimizer, actF=activeFunc, test=False).history['val_accuracy']
                    hiss = [a+b for a, b in zip(trained, hiss)]
                hiss=[h/r for h in hiss]
                legend.append([str(batchSize) + '-' + optimizer + '-' + activeFunc, hiss[-1]])
                hist.append(hiss)
                graph_NN([hist[-1]], [legend[-1][0]])
        with open(r"savedPNG/history.pickle", "wb") as file:
            pickle.dump([hist, legend], file)
    else:
        with open(r"savedPNG/history.pickle", "rb") as file:
            (hist, legend) = pickle.load(file)

    # graph all results
    sums=[l[1] for l in legend]
    legend=[l[0] for l in legend]

    graph_NN(hist, legend, save=True)

    # graph and print individual results
    for i in range(len(legend)):
        hist[i].append(legend[i])
        hist[i].append(sums[i])
    hist = sorted(hist, key=lambda row: row[-1], reverse=True)

    for h in hist:
        if h[-1]>=target:
            print(h[-2].split("-")[2:4])
            graph_NN([h[:][:-2]], [h[:][-2]], save=True)


if __name__ == '__main__':
    import pickle

    x,y, key=loadCSV(scale=True)
    layers=[
        [20, None],
        [30, None],
        [20, 'tanh'],
        [20, 'relu'],
        [10, 'relu'],
        [20, 'relu'],
        [10, 'relu'],
    ]

    test(x, y,
        layers,
        [0,4,5,8,10],
        [['Adamax', 'softplus'], ],
        20,
        r=15,
        load=False)
