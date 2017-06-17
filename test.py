import threading
import time
start = time.time()

def worker(arg):
    print "Worker %s" %arg

'''
for i in range(1000):
    t = threading.Thread(target=worker(i))
    t.start()
'''

for i in range(1000):
    print i

end = time.time()
print(end - start)