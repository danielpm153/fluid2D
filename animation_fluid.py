from simulation_fluid import *
import time
import matplotlib.animation
import matplotlib.pyplot
import numpy
import matplotlib
#matplotlib.use("TkAgg")  # somente no pycharm

performanceData = True

obj = MeshImage(
    '/image/barra2.png')

simu = FluidSimulation2D(obj)
simu.viscosity = 0.005
simu.u0 = 0.12
simu.reshape(add_up = 40, add_down = 40, add_left = 40, add_right = 150)
#simu.reshape(-30, -30, -30, -30)

simu.initialize()
height, width = simu.shape()
barrier = simu.obj
#print(simu.x_ref, simu.y_ref)
#38-68
#Re = simu.u0*3/simu.viscosity

theFig = matplotlib.pyplot.figure(figsize=(8, 3))

fluidImage = matplotlib.pyplot.imshow(simu.curl(), origin='lower', norm=matplotlib.pyplot.Normalize(-2*simu.u0, 2*simu.u0),
                                      cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')
bImageArray = numpy.zeros((height, width, 4), numpy.uint8)  # an RGBA image
bImageArray[barrier, 3] = 255								# set alpha=255 only at barrier sites

barrierImage = matplotlib.pyplot.imshow(bImageArray)

'''
height, widht = 100,100
barrier2 = numpy.zeros((height, width), bool)
barrier2[(height//2)-8:(height//2)+8, height //2] = True
bImageArray2 = numpy.zeros((height, width, 4), numpy.uint8)
bImageArray2[barrier2, 3] = 255
barrierImage2 = matplotlib.pyplot.imshow(bImageArray2)
'''

startTime = time.perf_counter()


def nextFrame(arg):							# (arg is the frame number, which we don't need)
    global startTime
    if performanceData and (arg % 100 == 0) and (arg > 0):
        #endTime = time.clock()
        endTime = time.perf_counter()
        print("%1.1f" % (100/(endTime-startTime)), 'frames per second')
        startTime = endTime
    #frameName = "frame%04d.png" % arg
    # matplotlib.pyplot.savefig(frameName)
    #frameList.write(frameName + '\n')
    for step in range(20):					# adjust number of steps for smooth animation
        simu.stream()
        simu.collide()

    # fluidImage.set_array(simu.curl())
    #print(simu.Fx, simu.Fy)
    #print(simu.Fx/(0.5*simu.u0**2*3*2))
    fluidImage.set_array(simu.ux)
    # return the figure elements to redraw
    return (fluidImage, barrierImage)#, barrierImage2)


animate = matplotlib.animation.FuncAnimation(
    theFig, nextFrame, interval=1, blit=True)
matplotlib.pyplot.show()
