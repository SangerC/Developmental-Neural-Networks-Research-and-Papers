import matplotlib.pyplot as plt

def fileToArray(fileName):
    arr = []
    time = 0
    with open(fileName) as f:
        for line in f:
            if line[0] == 'T':
                time = int(float(line.split(" ")[1]))
                break
            arr.append(float(line))
    return arr, time

def arrayToRollingAvg(array, size=20):
    start = 0
    end = size
    curr_total = sum(array[:size])
    out = [0 for i in range(size)]
    out.append(curr_total/size)

    while end < len(array) - 1:
        end += 1
        curr_total += array[end]
        curr_total -= array[start]
        start += 1
        out.append(curr_total/(size))

    return out
    
    

arr1 = fileToArray("baselineonehiddenlayer16nodropout")
arr2 = fileToArray("OneLinearLayer")
arr3 = fileToArray("aprilFigs/GrowthFrom2LayerWithPerformanceTriggerNoSelectiveTraining")
arr4 = fileToArray("aprilFigs/2linearSwitchTrainHalfwayResetOptimizer")

print(arrayToRollingAvg(arr1[0]))

plt.rcParams.update({'font.size': 20})
fig = plt.figure()
plt.plot(arrayToRollingAvg(arr1[0]), label=f"baseline 2 layer, Time: {arr1[1]}s")
plt.plot(arrayToRollingAvg(arr2[0]), label=f"baseline 3 layer, Time: {arr2[1]}s")
plt.plot(arrayToRollingAvg(arr3[0]), label=f"Grow no resest, Time: {arr3[1]}s")
plt.plot(arrayToRollingAvg(arr4[0]), label=f"Grow reset, Time: {arr3[1]}s")
plt.title("Add a Layer CartPole Task")
plt.ylabel("Time (Rolling Average 20 episodes)")
plt.xlabel("Episode")
plt.ylim(0,500)
plt.xlim(20)
spacing = .2
fig.subplots_adjust(bottom=spacing)
plt.legend()
plt.show()
