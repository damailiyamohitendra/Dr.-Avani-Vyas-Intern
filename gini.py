def test_split(index,value,dataset):
    left=[]
    right=[]

    for i in dataset:                           #this whole function is used to split the datset into groups
        if i[index]==value:
            left.append(i)
        else:
            right.append(i)
    return left,right

# left,right=test_split(0,"sunny",dataset)


#FUNCTION TO CALCULATE GINI VALUE
def gini(groups,classes):
    total_rows=sum(len(i) for i in groups) #calculate the total rows 
    gini=0

    for i in groups:
        size=len(i)
        if size==0:
            continue
        
        score=0.0
        label=[j[-1] for j in i]  # it gives the label of each row (weather yes or no) according to dataset

        for j in classes:
            x=label.count(j)/size # finding the value of each Pi
            score= score + x**2  #square of Pi

        gini += (1 - score) * (size / total_rows)  #  Gini formula

    
    return gini

def get_best_split(dataset):
        
        total_labels=list(set(i[-1] for i in dataset)) # give unique label
        best_index= None
        best_value=None
        best_score=float('inf')
        best_group=None

        for index in range(len(dataset[0])-1):  #it gives  the column from 0 to second last(we dont need label coloum )
            featured_value=set(i[index] for i in dataset)
            for value in featured_value:
                left,right=test_split(index,value,dataset)
                groups=[left,right]
                gini_score = gini(groups, total_labels)  # ✅
                if gini_score<best_score:
                    best_index= index
                    best_value=value
                    best_score=gini_score
                    best_group=groups
        
        return {
        'index': best_index,
        'value': best_value,
        'score': best_score,
        'groups': best_group
        }
                    


dataset = [
    ['Sunny', 'Yes'],
    ['Sunny', 'Yes'],
    ['Rainy', 'No'],
    ['Rainy', 'No'],
    ['Overcast', 'Yes']
]


best_split = get_best_split(dataset)


print("Best Feature Index to Split On:", best_split['index'])
print("Best Feature Value to Split On:", best_split['value'])
print("Gini Score of Best Split:", best_split['score'])

print("\nLeft Group (Matching Rows):")
for row in best_split['groups'][0]:
    print(row)

print("\nRight Group (Remaining Rows):")
for row in best_split['groups'][1]:
    print(row)
