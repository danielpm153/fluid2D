import numpy
import cv2


class MeshImage:
    def __init__(self, image_path: str) -> None:
        self.__image = cv2.imread(image_path)
        self.__image_gray = cv2.cvtColor(self.__image, cv2.COLOR_RGB2GRAY)
        self.__height = self.__image_gray.shape[0]
        self.__width = self.__image_gray.shape[1]
        self.gray_tone = 250

    def shape(self) -> tuple:
        '''
        return: height and width of the image
        '''
        return self.__height, self.__width

    def Mesh(self) -> numpy.ndarray:
        '''
        return: numpy array
        '''
        self.__meshObject = self.__gray_tone >= self.__image_gray
        return self.__meshObject

    @property
    def gray_tone(self) -> int:
        '''
        return: int
        '''
        return self.__gray_tone

    @gray_tone.setter
    def gray_tone(self, value: int) -> None:
        if 255 >= value >= 0:
            self.__gray_tone = value
        else:
            raise ValueError("The value of is between 0 and 255")


class FluidSimulation2D:
    def __init__(self, obj_mesh2D: MeshImage):
        self.obj_mesh2D = obj_mesh2D
        self.__height, self.__width = obj_mesh2D.shape()
        self.obj = self.obj_mesh2D.Mesh()

        self.__viscosity = 0.005
        self.__u0 = 0.12  # speed in x direction
        self.omega = 1 / (3*self.__viscosity + 0.5)

        # lattice-Boltzmann weight factors
        self.__w0 = 4/9
        self.__w1_4 = 1/9
        self.__w5_8 = 1/36

        # referencia mais à cima-direita(padrao da tela)
        self.x_ref, self.y_ref = 0, 0
        
        self.count = 0

    def reshape(self, add_up=0, add_down=0, add_right=0, add_left=0):
        '''
        :param add_up: concatenate layers of zero above the matrix
        :param add_down: concatenate zero layers below the matrix
        :param add_right: concatenate zero layers to the right of the matrix
        :param add_left: concatenate zero layers to the left of the matrix
        :return:
        '''
        if add_up > 0:
            zeros_width_up = numpy.zeros(
                (add_up, self.obj.shape[1]), dtype=bool)
            self.obj = numpy.concatenate((zeros_width_up, self.obj), axis=0)
        elif add_up < 0:
            self.obj = self.obj[-add_up:]

        if add_down > 0:
            zeros_width_down = numpy.zeros(
                (add_down, self.obj.shape[1]), dtype=bool)
            self.obj = numpy.concatenate((self.obj, zeros_width_down), axis=0)
        elif add_down < 0:
            self.obj = self.obj[:add_down]

        if add_right > 0:
            zeros_height_right = numpy.zeros(
                (self.obj.shape[0], add_right), dtype=bool)
            self.obj = numpy.concatenate(
                (self.obj, zeros_height_right), axis=1)
        elif add_right < 0:
            self.obj = self.obj[:, -add_right:]

        if add_left > 0:
            zeros_height_left = numpy.zeros(
                (self.obj.shape[0], add_left), dtype=bool)
            self.obj = numpy.concatenate((zeros_height_left, self.obj), axis=1)
        elif add_left < 0:
            self.obj = self.obj[:, :add_left]

        self.__height, self.__width = self.obj.shape

        if self.x_ref+add_left >= 0:
            self.x_ref += add_left
        if self.y_ref+add_up >= 0:
            self.y_ref += add_up

    def shape(self) -> tuple:
        '''
        return: height and width of the mesh
        '''
        return self.__height, self.__width

    def initialize(self):
        '''
        Initialize all arrays to stabilize the flow to the right
        particle densities over 9 directions
        '''
        self.Fx, self.Fy = 0, 0
        
        self.n0 = self.__w0 * \
            (numpy.ones((self.__height, self.__width)) - 1.5*self.__u0**2)
        self.nN = self.__w1_4 * \
            (numpy.ones((self.__height, self.__width)) - 1.5*self.__u0**2)
        self.nS = self.__w1_4 * \
            (numpy.ones((self.__height, self.__width)) - 1.5*self.__u0**2)
        self.nE = self.__w1_4 * (numpy.ones((self.__height, self.__width)) +
                                 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nW = self.__w1_4 * (numpy.ones((self.__height, self.__width)) -
                                 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nNE = self.__w5_8 * (numpy.ones((self.__height, self.__width)) +
                                  3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nSE = self.__w5_8 * (numpy.ones((self.__height, self.__width)) +
                                  3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nNW = self.__w5_8 * (numpy.ones((self.__height, self.__width)) -
                                  3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nSW = self.__w5_8 * (numpy.ones((self.__height, self.__width)) -
                                  3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        # macroscopic properties
        self.rho = self.n0 + self.nN + self.nS + self.nE + \
            self.nW + self.nNE + self.nSE + self.nNW + self.nSW
        self.ux = (self.nE + self.nNE + self.nSE -
                   self.nW - self.nNW - self.nSW) / self.rho
        self.uy = (self.nN + self.nNE + self.nNW -
                   self.nS - self.nSE - self.nSW) / self.rho

        # object initialization, true where there is part of the object
        #self.obj = self.obj_mesh2D.Mesh()
        self.objN = numpy.roll(self.obj,  1, axis=0)
        self.objS = numpy.roll(self.obj, -1, axis=0)
        self.objE = numpy.roll(self.obj,  1, axis=1)
        self.objW = numpy.roll(self.obj, -1, axis=1)
        self.objNE = numpy.roll(self.objN,  1, axis=1)
        self.objNW = numpy.roll(self.objN, -1, axis=1)
        self.objSE = numpy.roll(self.objS,  1, axis=1)
        self.objSW = numpy.roll(self.objS, -1, axis=1)

    @property
    def viscosity(self):
        return self.__viscosity

    @viscosity.setter
    def viscosity(self, value: float):
        if value > 0:
            self.__viscosity = value
            self.omega = 1 / (3*self.__viscosity + 0.5)
        else:
            raise ValueError

    @property
    def u0(self):
        return self.__u0

    @u0.setter
    def u0(self, value: float):
        self.__u0 = value

    def stream(self):
        # axis 0 is north-south; + direction is north
        self.nN = numpy.roll(self.nN,   1, axis=0)
        self.nNE = numpy.roll(self.nNE,  1, axis=0)
        self.nNW = numpy.roll(self.nNW,  1, axis=0)
        self.nS = numpy.roll(self.nS,  -1, axis=0)
        self.nSE = numpy.roll(self.nSE, -1, axis=0)
        self.nSW = numpy.roll(self.nSW, -1, axis=0)
        # axis 1 is east-west; + direction is east
        self.nE = numpy.roll(self.nE,   1, axis=1)
        self.nNE = numpy.roll(self.nNE,  1, axis=1)
        self.nSE = numpy.roll(self.nSE,  1, axis=1)
        self.nW = numpy.roll(self.nW,  -1, axis=1)
        self.nNW = numpy.roll(self.nNW, -1, axis=1)
        self.nSW = numpy.roll(self.nSW, -1, axis=1)
        # Use tricky boolean arrays to handle obj collisions (bounce-back):
        self.nN[self.objN] = self.nS[self.obj]
        self.nS[self.objS] = self.nN[self.obj]
        self.nE[self.objE] = self.nW[self.obj]
        self.nW[self.objW] = self.nE[self.obj]
        self.nNE[self.objNE] = self.nSW[self.obj]
        self.nNW[self.objNW] = self.nSE[self.obj]
        self.nSE[self.objSE] = self.nNW[self.obj]
        self.nSW[self.objSW] = self.nNE[self.obj]

        #height_origin, width_origin = self.obj_mesh2D.Mesh
        
        self.Fx = sum(sum((self.nE + self.nNE + self.nSE -
                           self.nW - self.nNW - self.nSW)*self.obj))
        self.Fy = sum(sum((self.nN + self.nNE + self.nNW -
                           self.nS - self.nSE - self.nSW)*self.obj))
        
        

    def collide(self):
        self.rho = self.n0 + self.nN + self.nS + self.nE + \
            self.nW + self.nNE + self.nSE + self.nNW + self.nSW
        self.ux = (self.nE + self.nNE + self.nSE -
                   self.nW - self.nNW - self.nSW) / self.rho
        self.uy = (self.nN + self.nNE + self.nNW -
                   self.nS - self.nSE - self.nSW) / self.rho
        ux2 = self.ux**2
        uy2 = self.uy**2
        u2 = ux2 + uy2
        omu215 = 1 - 1.5*u2
        uxuy = self.ux * self.uy

        self.n0 = (1-self.omega)*self.n0 + self.omega * \
            self.__w0 * self.rho * omu215
        self.nN = (1-self.omega)*self.nN + self.omega * \
            self.__w1_4 * self.rho * (omu215 + 3*self.uy + 4.5*uy2)
        self.nS = (1-self.omega)*self.nS + self.omega * \
            self.__w1_4 * self.rho * (omu215 - 3*self.uy + 4.5*uy2)
        self.nE = (1-self.omega)*self.nE + self.omega * \
            self.__w1_4 * self.rho * (omu215 + 3*self.ux + 4.5*ux2)
        self.nW = (1-self.omega)*self.nW + self.omega * \
            self.__w1_4 * self.rho * (omu215 - 3*self.ux + 4.5*ux2)

        self.nNE = (1-self.omega)*self.nNE + self.omega * self.__w5_8 * \
            self.rho * (omu215 + 3*(self.ux+self.uy) + 4.5*(u2+2*uxuy))
        self.nNW = (1-self.omega)*self.nNW + self.omega * self.__w5_8 * \
            self.rho * (omu215 + 3*(-self.ux+self.uy) + 4.5*(u2-2*uxuy))
        self.nSE = (1-self.omega)*self.nSE + self.omega * self.__w5_8 * \
            self.rho * (omu215 + 3*(self.ux-self.uy) + 4.5*(u2-2*uxuy))
        self.nSW = (1-self.omega)*self.nSW + self.omega * self.__w5_8 * \
            self.rho * (omu215 + 3*(-self.ux-self.uy) + 4.5*(u2+2*uxuy))

        # Force steady rightward flow at ends (no need to set 0, N, and S components):
        self.nE[:, 0] = self.__w1_4 * \
            (1 + 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nW[:, 0] = self.__w1_4 * \
            (1 - 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nNE[:, 0] = self.__w5_8 * \
            (1 + 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nSE[:, 0] = self.__w5_8 * \
            (1 + 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nNW[:, 0] = self.__w5_8 * \
            (1 - 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)
        self.nSW[:, 0] = self.__w5_8 * \
            (1 - 3*self.__u0 + 4.5*self.__u0**2 - 1.5*self.__u0**2)

    def curl(self):
        return numpy.roll(self.uy, -1, axis=1) - numpy.roll(self.uy, 1, axis=1) - numpy.roll(self.ux, -1, axis=0) + numpy.roll(self.ux, 1, axis=0)


if __name__ == "__main__":
    obj = MeshImage(
        '/home/daniel/Área de Trabalho/python_codigos/fluidos/simu 2D/image/circulo3.png')
    print(type(obj.shape()))
    simu = FluidSimulation2D(obj)
    simu.initialize()
    for i in range(50):
        simu.stream()
        simu.collide()
    ux2 = simu.ux
    uy2 = simu.uy
