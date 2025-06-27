Dataset=[
    #squares in feet , house price in million
    [1300,240],
    [1500,320],
    [1700,330],
    [1830,295],
    [1550,256],
    [2350,409],
    [1450,319]
 
]
def value(dataset):
    #mean of x
    count=0
    total=0
    for i in dataset:
        count=count+i[0]
        total+=1
    mean_x=count/total
    # print(mean_x)


    # mean of y
    count_y=0
    for i in dataset:
        count_y=count_y+i[1]

    mean_y=count_y/total
    # print(mean_y)

    total = len(dataset)

   

    
    #deviation of x 
    dev_x=[]
    dev_y=[]
    for i in dataset:
        a = i[0] - mean_x
        dev_x.append(a)
        b = i[1]-mean_y
        dev_y.append(b)
    
   # print(dev_x,dev_y)


    #product of deviation

    product_list = [a * b for a, b in zip(dev_x, dev_y)]
        
    

    sum_of_deviation=sum(product_list)

    #square of deviation x
    sq_dev_x=[]
    for i in dev_x:
        num=i*i
        sq_dev_x.append(num)
    
    sum_sq_deviation=sum(sq_dev_x)
    
    #calculate m 
    m= sum_of_deviation/sum_sq_deviation

    #calculate b
    b= mean_y-(m*mean_x)

    return m,b

a,b=value(Dataset)
print(a)
print(b)

def predict(dataset,a,b,input):
    z=(input*a)+b
    print(z)


input=1450
predict(Dataset,a,b,input)