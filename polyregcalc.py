import numpy
import matplotlib.pyplot as plt
from random import randint
from time import sleep

def rand():
  return(randint())

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 9))

line = numpy.linspace(1, 22, 100)



plt.scatter(x, y)
plt.plot(line, mymodel(line))
plt.show()

print(mymodel)

while True:
  for x in range(50):
    print(mymodel(x))
    sleep(0.2)
