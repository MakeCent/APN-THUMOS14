start = []
end = []
start = []
end = []
for v, p in predictions.items():
    p = ordinal2completeness(p)
    end.extend(p[ground_truth[v][:,1]])
    start.extend(p[ground_truth[v][:,0]])
from matplotlib import pyplot as plt
plt.boxplot([start, end])
plt.title("unweighted, 50-21.79")
plt.show()