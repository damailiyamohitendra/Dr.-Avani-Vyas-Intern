# GIVEN DATA
data=[
    [1,2],[2,3],[1,4],[1.4,2.1],
    [1.5,4.5],[2.6,1.4]
]

n= len(data)

# MANHATTAN FORMULA FOR FINDING DISTANCE MATRIX
def diff_formula(point1,point2):
    a= abs(point1[0]-point2[0])
    b=abs(point1[1]-point2[1])
    c=a+b
    return c


labels=['A','B','C','D','E','F']
distance=[]


# DISTANCE MATRIX 
for i in range(n):
    row=[]
    for j in range(n):
        row.append(diff_formula(data[i],data[j]))
    distance.append(row)


# for printing distance matrix
# for j in distance:
#     print(j)

clusters=[]

# it will add items labels in clusetr (initially each point is considered as label)
for i in labels:
    clusters.append([i])

def single_link(cluster1,cluster2,distance,labels):
    min_dist=float('inf')
    for i in cluster1:
        for j in cluster2:
            index1=labels.index(i)
            index2=labels.index(j)

            dist=distance[index1][index2]

            if dist<min_dist:
                min_dist=dist
    return min_dist

while len(clusters)>1:
    min_distance=float('inf')

    first=None
    second=None

    for i in range(len(clusters)):
        for j in range(i+1,len(clusters)):
            cluster1=clusters[i]
            cluster2=clusters[j]

            dist=single_link(cluster1,cluster2,distance,labels)

            if dist<min_distance:
                min_distance=dist
                first=i
                second=j

    # print(first)
    # print(second)
    # print(min_distance)

    new_cluster=clusters[first]+clusters[second]

    if first<second:
        clusters.pop(second)
        clusters.pop(first)
    else:
        clusters.pop(first)
        clusters.pop(second)
    
    clusters.append(new_cluster)

    print(clusters)


    