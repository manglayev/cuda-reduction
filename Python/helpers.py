import reduction_1

def initializeArray(CUDASIZE):
    return [1] * CUDASIZE

def checkResults(array):
    sum = 0
    for element in array:
        sum += element
    return sum

def callReduction[BLOCKS, THREADS](VARIANT, dev_a, dev_b):
    if VARIANT == 1:
        reduction_1.reduction_1(dev_a, dev_b)
'''
    if VARIANT == 7:
        cuda_global[BLOCKS, THREADS](dev_a, dev_b)
'''
