from project4_2.NN_Base import *


def count_y():
    from matplotlib import pyplot as plt
    from random import random
    data_list, path = loadRawInfo("Train_data")
    counts=[0]*5
    multiplier={0:1.2, 1:4, 2:3, 3:6, 4:1}
    for k in range(1000):
        for s in data_list:
            loop_count = 1#multiplier[s[3]]
            if random() < loop_count - round(loop_count): loop_count += 1
            loop_count = round(loop_count)
            for j in range(loop_count):
                counts[s[3]]+=1


    l={0: 'Angry', 4: 'Neutr', 1: 'Happy', 2: 'Sad', 3: 'Wonde'}
    multiplier={0:1, 1:5.2, 2:2.15, 3:4.4, 4:1}
    for i,s in enumerate(counts): print(l[i], counts[i]/1000, "|", round(counts[i]/sum(counts)*100, 3), "%")
    print(sum(counts)/1000)
    l=["Angry", "Neutr", "Happy", 'Sad', 'Wonde']
    plt.figure(figsize=(9, 3))
    plt.bar(l, counts)
    plt.suptitle('Categorical Plotting')
    plt.show()


def count_y_np():
    from matplotlib import pyplot as plt
    train_x, train_y, train_ID, test_x, test_y, test_ID = load_or_clean_data(raw=False)
    counts=[0]*5
    for s in test_y: counts[s.argmax()]+=1

    l={0: 'Angry', 4: 'Neutr', 1: 'Happy', 2: 'Sad', 3: 'Wonde'}
    for i,s in enumerate(counts): print(l[i], counts[i], "|", round(counts[i]/sum(counts)*100, 3), "%")
    print(sum(counts))
    l=["Angry", "Neutr", "Happy", 'Sad', 'Wonde']
    plt.figure(figsize=(9, 3))
    plt.bar(l, counts)
    plt.suptitle('Categorical Plotting')
    plt.show()


if __name__ == '__main__':
    # count_y()
    count_y_np()
