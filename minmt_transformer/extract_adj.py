import numpy
import time

window_size = 0
name_dir = "xxx"
count = 0
t0 = time.time()
output_file = open(name_dir+"xxx.adj","w")
with open(name_dir+"src1.bpe") as f_src1, \
     open(name_dir+"src2.bpe") as f_src2, \
     open(name_dir+"src.align") as align_file:  
     if True:
         for en,fr, line in zip(f_src1,f_src2, align_file):
            count += 1
            if count % 100 == 0:
                t1 = time.time()
                t = t1 - t0
                t0 = time.time()
                print(str(count)+": "+str(t)+"s")
            l1 = len(en.split())+1
            l2 = len(fr.split())+1
            adj = numpy.zeros((l1,l2))
            line = line.split()
            for v in line:
                v = v.split("-")
                v1 = int(v[0])
                v2 = int(v[1])
                adj[v1][v2] = 1.
                for w in range(window_size):
                    w = w+1
                    if v1+w < l1 :
                        adj[v1+w][v2] = 1.
                    if v2+w < l2 :
                        adj[v1][v2+w] = 1.
                    if v1-w >= 0 :
                        adj[v1-w][v2] = 1.
                    if v2-w >=0 :
                        adj[v1][v2-w] = 1.
            adj[l1-1][l2-1]=1.
            for i in adj.flat:
                output_file.write(str(float(i))+" ")
            output_file.write("\n") 
print("END!")
