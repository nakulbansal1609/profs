import random
import pandas as pd




def random_sample(input,secret=''):
    count = 1
    start = 1
    stop = 100000000000000
    step = 1

    #Generate a unique integer based on the string input
    #Use your own algo and try to make not very long
    if secret=='':
        random.seed(input)
    else:
        ascii_secret = sum([ord(c) for c in secret])+len(secret)
        #Calculate the cantor output
        cantor_output = int(0.5*(input + ascii_secret)*(input + ascii_secret + 1) + input)
        random.seed(cantor_output)

    def gen_random():
        while True:
            yield random.randrange(start, stop, step)

    def gen_n_unique(source, n):
        seen = set()
        seenadd = seen.add
        for i in (i for i in source() if i not in seen and not seenadd(i)):
            yield i
            if len(seen) == n:
                break

    return [i for i in gen_n_unique(gen_random,
                                    min(count, int(abs(stop - start) / abs(step))))][0]


#Example
secret = 'AktNr'
random_sample(1, secret) #94456427138402
random_sample(2, secret) #39893616031233
random_sample(3, secret) #41631129812209

secret2 = 'GldNr'
random_sample(1, secret2) #42722234574275
random_sample(2, secret2) #97722417962347
random_sample(3, secret2) #48910014941716

df = pd.DataFrame(columns=['input','output'])
for i in range(100000):
    row = {'input':i,'output':random_sample(i)}
    df = df.append(row, ignore_index=True)

df.to_csv('randon_numbers.csv',index=False)
