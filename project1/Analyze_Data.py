from pandas import read_csv

# graph size and amount of people by attribute "colName" in relation to "testing"
def average(colName, testing, sub="", save=True):
    nums={}
    if sub != "":
        maximum = 0
        minimum = 10000000000000000000000
        for st in dataset[colName]:
            maximum = max(maximum, st)
            minimum = min(minimum, st)
        sub = int((minimum + maximum) / sub + 1)

        nums = {}
        for j in range(int(minimum), int(maximum + 1), sub):
            nums[j] = [0, 0]

        for i, st in enumerate(dataset[colName]):
            for j in range(int(minimum), int(maximum + 1), sub):
                if j <= st < j + sub:
                    break
            nums[j][0] += 1
            nums[j][1] += dataset[testing][i]
    else:
        for i, st in enumerate(dataset[colName]):
            if not(st in nums.keys()):
                nums[st] = [0,0]
            nums[st][0]+=1
            nums[st][1]+=dataset[testing][i]

    # print the results
    for a in list(nums):
        if nums[a][0]==0: nums.pop(a); continue
        nums[a]=[nums[a][1]/nums[a][0]*100, nums[a][0]]
    nums=dict(sorted(nums.items()))

    print(colName, sub)
    for key in nums.keys():
        print(key, "#"+str(round(nums[key][1]/100, 1))+"% :", round(nums[key][0], 1))
    print()

    # graph the results
    data = {'group': [], testing: [], 'count': [], }
    for key in nums.keys():
        if sub != "": data['group'].append(key+sub/2)
        else: data['group'].append(key)
        data[testing].append(nums[key][0])
        data['count'].append(nums[key][1]*len(nums)/20)

    import matplotlib.pyplot as plt
    if not(type(data['group'][0]) is str) and sub == "": plt.xticks(data['group'])
    plt.scatter('group', testing, c='#5f7f7f', s='count', data=data)
    plt.plot('group', testing, data=data, linewidth=0.25)
    plt.ylim(bottom=min(min(data[testing])-3, 10), top=max(min(max(data[testing])+5,100), 30))
    plt.xlabel(colName)
    plt.ylabel('% ' + testing)
    if save: plt.savefig(r"savedPNG/ "+colName+testing)
    plt.show()


dataset = read_csv("data.csv", usecols=["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                                        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"])

average("Gender", "Exited")
average("Geography", "Exited")
average("Age", "Exited", 20)
average("Balance", "Exited", 20)
average("EstimatedSalary", "Exited", 30)
average("CreditScore", "Exited", 20)
average("Tenure", "Exited")
average("NumOfProducts", "Exited")
average("HasCrCard", "Exited")
average("IsActiveMember", "Exited")