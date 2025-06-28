import copy 

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


def find_max_avg_row(distance):
    max_avg = float('-inf')
    max_index = -1

    for i in range(len(distance)):
        avg = sum(distance[i]) / len(distance[i])
        if avg > max_avg:
            max_avg = avg
            max_index = i

    return max_index

def split_matrix_by_index(distance, labels, index):
    picked_label = [labels[index]]
    remaining_labels = labels[:index] + labels[index+1:]

    new_matrix = []
    for i in range(len(distance)):
        if i == index:
            continue
        row = distance[i][:index] + distance[i][index+1:]
        new_matrix.append(row)

    return picked_label, remaining_labels, new_matrix

def divisive_clustering(distance,labels):
    matrix=distance
    matrix_label=labels.copy()
    seprated=[]

    while  len(matrix)>1:
        index=find_max_avg_row(matrix)
        seprated_label,remaining_label,new_matrix=split_matrix_by_index(matrix,matrix_label,index)
        
        seprated.append([seprated_label,remaining_label])

        print(f"{seprated_label} and {remaining_label}")

        matrix_label= remaining_label
        matrix=new_matrix

    return seprated

divisive_clustering(distance, labels)