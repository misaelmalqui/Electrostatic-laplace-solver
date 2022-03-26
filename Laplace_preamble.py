# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:01:12 2021

@author: misae
"""

import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import scipy.interpolate
from scipy import integrate
from shapely.geometry import Point, Polygon


# Graphs configuration

import locale
locale.setlocale(locale.LC_NUMERIC, "ca_ES")

plt.rcParams.update({
    "text.usetex": True,
    })
pre = r"""  \usepackage{amsmath}
            \usepackage{amssymb}
            \usepackage[decimalsymbol=comma]{siunitx}"""
plt.rc('text.latex', preamble=pre)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.formatter.use_locale'] = True


# Potential distributions

class FullCircle:
    def __init__(self, x, y, r1, r2, val, n=8):
        """
        Full Circle.

        Parameters
        ----------
        x : float
            x component.
        y : float
            y component.
        val : float
            Extra potential with respect the intial potential of the media.
        r1 : float, optional
            Radius for E.Field, P.Field, and the plot.
        r2 : float, optional
            Radius for the surf() and is_close() methods.
        n : int, optional
            Number of field lines that the object generates.

        Raises
        ------
        ValueError
            Error if r1 >= r2.

        Returns
        -------
        None.

        """
        if r1 >= r2:
            raise ValueError('r2 must be greater than r1.')

        self.x = x
        self.y = y
        self.r1 = r1
        self.r2 = r2
        self.val = val
        self.sign = np.sign(val)

        angles = np.linspace(0, 2*np.pi*(n-1)/n, n)
        self.surf = [[x + r2*np.cos(alpha),
                      y + r2*np.sin(alpha)] for alpha in angles]

    def condition(self, x, y):
        """
        Parameters
        ----------
        x : np.meshgrid
            x component.
        y : np.meshgrid
            y component.

        Returns
        -------
        np.array
            Returns a matrix where the points that are in the object have the
            value of the object.

        """
        return np.select([(x-self.x)**2 + (y-self.y)**2 <= self.r1**2,
                          (x-self.x)**2 + (y-self.y)**2 > self.r1**2],
                         [True, False])


    def is_close(self, x, y):
        """
        Parameters
        ----------
        x : float
            x component.
        y : float
            y component.

        Returns
        -------
        boolean
            Returns True if (x,y) is close enough to the object and False
            otherwise.

        """
        return np.sqrt((x - self.x)**2 + (y - self.y)**2) <= self.r2

    def plot(self):
        """
        Generates a plot of the wire, a circle of radius r1.

        Returns
        -------
        None.

        """
        diccolors = {'1': 'r', '0': 'k', '-1': 'b'}
        figure = plt.Circle((self.x, self.y), self.r1,
                            color=diccolors[str(self.sign)], zorder=10)
        plt.gca().add_artist(figure)
        return None

class Rectangle:
    def __init__(self, x, y, lx, ly, w, angle, val, nx, ny):
        self.x = x
        self.y = y
        self.val = val
        self.lx = lx
        self.ly = ly
        self.w = w
        self.angle = angle
        self.val = val
        self.sign = np.sign(val)
        
        
        q1x = lx/2*np.cos(angle) - ly/2*np.sin(angle) + self.x
        q1y = lx/2*np.sin(angle) + ly/2*np.cos(angle) + self.y
        
        q2x = - lx/2*np.cos(angle) - ly/2*np.sin(angle) + self.x
        q2y = - lx/2*np.sin(angle) + ly/2*np.cos(angle) + self.y
        
        q3x = - lx/2*np.cos(angle) + ly/2*np.sin(angle) + self.x
        q3y = - lx/2*np.sin(angle) - ly/2*np.cos(angle) + self.y
        
        q4x = lx/2*np.cos(angle) + ly/2*np.sin(angle) + self.x
        q4y = lx/2*np.sin(angle) - ly/2*np.cos(angle) + self.y
        
        r1x = (lx/2 + w)*np.cos(angle) - (ly/2 + w)*np.sin(angle) + self.x
        r1y = (lx/2 + w)*np.sin(angle) + (ly/2 + w)*np.cos(angle) + self.y
        
        r2x = - (lx/2 + w)*np.cos(angle) - (ly/2 + w)*np.sin(angle) + self.x
        r2y = - (lx/2 + w)*np.sin(angle) + (ly/2 + w)*np.cos(angle) + self.y
        
        r3x = - (lx/2 + w)*np.cos(angle) + (ly/2 + w)*np.sin(angle) + self.x
        r3y = - (lx/2 + w)*np.sin(angle) - (ly/2 + w)*np.cos(angle) + self.y

        r4x = (lx/2 + w)*np.cos(angle) + (ly/2 + w)*np.sin(angle) + self.x
        r4y = (lx/2 + w)*np.sin(angle) - (ly/2 + w)*np.cos(angle) + self.y
        
        self.q3x = q3x
        self.q3y = q3y
        
        self.pol1 = Polygon([(q1x, q1y), (q2x, q2y), (q3x, q3y), (q4x, q4y)])
        self.pol2 = Polygon([(r1x, r1y), (r2x, r2y), (r3x, r3y), (r4x, r4y)])
        
        self.surf = []
        if nx == 1:
            ss1 = [[(r1x + r2x)/2, (r1y + r2y)/2]]
            ss3 = [[(r3x + r4x)/2, (r3y + r4y)/2]]
        else:
            ss1 = [[r1x + (r2x - r1x)*i/(nx - 1), 
                    r1y + (r2y - r1y)*i/(nx - 1)] for i in range(nx)]
            ss3 = [[r3x + (r4x - r3x)*i/(nx - 1), 
                    r3y + (r4y - r3y)*i/(nx - 1)] for i in range(nx)]

        ss2 = [[r2x + (r3x - r2x)*(i + 1)/(ny + 1), 
                r2y + (r3y - r2y)*(i + 1)/(ny + 1)] for i in range(ny)]
        ss4 = [[r4x + (r1x - r4x)*(i + 1)/(ny + 1), 
                r4y + (r1y - r4y)*(i + 1)/(ny + 1)] for i in range(ny)]
            
        for i in range(nx):
            self.surf.append(ss1[i])
            self.surf.append(ss3[i])
            
        for i in range(ny):
            self.surf.append(ss2[i])
            self.surf.append(ss4[i])
    
    def condition_ind(self, x, y):
        return Point(x, y).within(self.pol1)
    
    def condition(self, x, y):
        cfunc = np.vectorize(self.condition_ind)
        return cfunc(x, y)
    
    def is_close(self, x, y):
        return Point(x, y).within(self.pol2)
    
    def plot(self):
        diccolors = {'1': 'r', '0': 'k', '-1': 'b'}

        figure = plt.Rectangle([self.q3x, self.q3y], self.lx, self.ly, 
                               facecolor=diccolors[str(self.sign)], 
                               angle=np.rad2deg(self.angle), zorder=10)
        plt.gca().add_artist(figure)
        return None

# General objects

class Grid:
    def __init__(self, xmin, xmax, ymin, ymax, prec):
        """
        Contains the arrays of x,y components for the plots

        Parameters
        ----------
        xmin : float
            lower x limit.
        xmax : float
            upper x limit.
        ymin : float
            lower y limit.
        ymax : float
            upper y limit.
        prec : float, optional
            Precisions, size of the cells of the matrix. The default is 0.001.

        Returns
        -------
        None.

        """
        dx = xmax - xmin
        dy = ymax - ymin
        dimx = int(dx/prec)
        dimy = int(dy/prec)

        self.xaxis1 = np.linspace(xmin, xmax, dimx)
        self.yaxis1 = np.linspace(ymin, ymax, dimy)
        self.x1, self.y1 = np.meshgrid(self.xaxis1, self.yaxis1)

        self.xaxis2 = np.linspace(xmin, xmax, dimx + 1)
        self.yaxis2 = np.linspace(ymin, ymax, dimy + 1)
        self.x2, self.y2 = np.meshgrid(self.xaxis2, self.yaxis2)


class Vfield:
    """Currently only works if the fixed potential points have a different
    value from the initial potential."""

    def __init__(self, charges, meshx, meshy, stdV):
        """
        Parameters
        ----------
        charges : list
            Contains the charge distributions for the system.
        meshx : np.meshgrid
            Grid of x components.
        meshy : np.meshgrid
            Grid of y components.
        stdV : float, optional
            Initial potential of the background.

        Returns
        -------
        None.

        """
        self.charges = charges

        self.matrix = stdV*np.ones(np.shape(meshx))
        self.matrix_bool = np.full(np.shape(meshx), False)
        for charge in charges:
            mi = charge.condition(meshx, meshy)
            mib = mi != 0
            self.matrix_bool = self.matrix_bool + mib
            self.matrix[mib] = charge.val


class Efield:
    def __init__(self, V, x_axis, y_axis, h_width, h_length, charges):
        """
        Matrix with x and y components of E.field.

        Parameters
        ----------
        V : np.array
            Matrix of potential field values.
        x_axis : np.meshgrid
            x component.
        y_axis : np.meshgrid
            y component.
        h_width : float
            width of arrow's head.
        h_length : float
            lemgth of arrow's head.
        charges : np.array
            Contains the charge distributions for the system.

        Returns
        -------
        None.

        """
        self.y, self.x = np.gradient(-V)
        self.t = np.sqrt(self.x**2 + self.y**2) + 0.0001
        self.dirx = np.divide(self.x[:, :], self.t[:, :])
        self.diry = np.divide(self.y[:, :], self.t[:, :])

        self.dirxfunc = scipy.interpolate.interp2d(x_axis, y_axis,
                                                   self.dirx, kind='quintic')
        self.diryfunc = scipy.interpolate.interp2d(x_axis, y_axis,
                                                   self.diry, kind='quintic')

        self.h_width = h_width
        self.h_length = h_length
        self.charges = charges

    def func(self, t, rp):
        """
        Parameters
        ----------
        t : float
            Time variable for the line method.
        rp : np.array
            List of x and y components of a point.

        Returns
        -------
        list
            Contains the functions for x and y components of the E.field.

        """
        x, y = rp
        return [self.dirxfunc(x, y)[0], self.diryfunc(x, y)[0]]

    def line(self, surface, limits, sign, startarrow=True, endarrow=True):
        """
        Plots the field lines starting from some given points.

        Parameters
        ----------
        surface : np.array
            List of [[x_i, y_i]] points to start the field lines.
        limits : np.array
            Contains the x and y lower and upper bounds.
        sign : float
            Direction to integrate the line.
        startarrow : boolean, optional
            If True the plot contains arrow heads at the beginning of each
            line. The default is True.
        endarrow : boolean, optional
            If True the plot contains arrow heads at the end of each line.
            The default is True.

        Returns
        -------
        None.

        """
        xmin = limits[0] + 0.0008
        xmax = limits[1] - 0.0008
        ymin = limits[2] + 0.0008
        ymax = limits[3] - 0.0008

        maxiter = 30000

        for dt in [-0.001, 0.001]:

            for i in range(int(len(surface))):
                r = integrate.ode(self.func)
                r.set_integrator('vode')
                x = [surface[i][0]]
                y = [surface[i][1]]
                r.set_initial_value([x[0], y[0]], 0)
                j = 0
    
                while r.successful():
                    r.integrate(r.t+dt)
                    x.append(r.y[0])
                    y.append(r.y[1])
                    j = j + 1
    
                    flag = False
                    for ch in self.charges:
                        if ch.is_close(r.y[0], r.y[1]):
                            flag = True
                            break
                    if r.y[0] > xmax or r.y[0] < xmin or r.y[1] > ymax or r.y[1] < ymin:
                        break
                    elif j >= maxiter or flag:
                        break
    
                plt.plot(x, y, color='k', linewidth=1)
    
                if dt < 0:
                    x.reverse()
                    y.reverse()
    
                if startarrow and (len(x) > maxiter/10):
                    n = int(len(x)/5)
                    dx = (x[n+1] - x[n])/100
                    dy = (y[n+1] - y[n])/100
                    plt.arrow(x[n], y[n], dx, dy, color='k',
                              head_width=self.h_width, head_length=self.h_length)
                if endarrow and (len(x) > maxiter/10):
                    n = int(len(x)/5)
                    dx = (x[-n+1] - x[-n])/100
                    dy = (y[-n+1] - y[-n])/100
                    plt.arrow(x[-n], y[-n], dx, dy, color='k',
                              head_width=self.h_width, head_length=self.h_length)
        return None

    def lines(self, limits):
        """
        Applies the line() method for all charge configurations using their
        default inputs.

        Parameters
        ----------
        limits : np.array
            Contains the x and y lower and upper bounds.

        Returns
        -------
        None.

        """
        for ch in self.charges:
            self.line(ch.surf, limits, ch.sign)
        return None


# General functions


def setbound(V, Vbool, t, b, l, r):
    """
    Function to set te boundary conditions of the plane.

    Parameters
    ----------
    V : np.array
        Matrix of V initial values.
    Vbool : np.array
        Matrix of U boolean values, 1 if the cell is inside an object and 0
        otherwise.
    t : array
        Top boundary condition.
    b : array
        Bottom boundary condition.
    l : array
        Left boundary condition.
    r : array
        Right boundary condition.

    Raises
    ------
    ValueError
        Error if the dimensions of the conditions lists do not coincide with
        the dimensions of the matrix.

    Returns
    -------
    None.

    """
    if (len(t) != len(V[0])) or (len(b) != len(V[0])):
        raise ValueError('Top and bottom conditions must have the same dimen'
                         'sions that Ux.')
    if (len(l) != len(V[:,0])) or (len(r) != len(V[:,0])):
        raise ValueError('Left and right conditions must have the same dimen'
                         'sions that Uy.')
    Vbool[0,:] = True
    Vbool[-1,:] = True
    Vbool[:,0] = True
    Vbool[:,-1] = True
    
    V[0,:] = b
    V[-1,:] = t
    V[:,0] = l
    V[:,-1] = r
    
    return None


@nb.jit(nopython=True, nogil=True)
def iteration(V, Vbool, iterations=10000):
    """
    Function to solve Laplace equation by the FDM.

    Parameters
    ----------
    V : np.array
        Matrix of V initial values.
    Vbool : np.array
        Matrix of U boolean values, 1 if the cell is inside an object and 0
        otherwise.
    iterations : float, optional
        Number of iterations for the loop. The default is 10000.

    Returns
    -------
    Mone.

    """
    for k in range(iterations):
        for x in range(len(V[0])):
            for y in range(len(V)):
                if Vbool[y, x]:
                    continue
                else:
                    newV = 0.
                    count = 0
                    if (x+1 >= 0 and x+1 < len(V[0])) and (y >= 0 and y < len(V)):
                        newV += V[y, x+1]
                        count += 1
                    if (x-1 >= 0 and x-1 < len(V[0])) and (y >= 0 and y < len(V)):
                        newV += V[y, x-1]
                        count += 1
                    if (x >= 0 and x < len(V[0])) and (y+1 >= 0 and y+1 < len(V)):
                        newV += V[y+1, x]
                        count += 1
                    if (x >= 0 and x < len(V[0])) and (y-1 >= 0 and y-1 < len(V)):
                        newV += V[y-1, x]
                        count += 1
                    V[y][x] = float(newV)/count
    return None


def autoplot(charges, x1, y1, x2, y2, V, lvl=10):
    """
    Makes the standard potential plot

    Parameters
    ----------
    charges : list
        Contains the potential distributions for the system.
    x1 : np.meshgrid
        Grid of x components for the contour plot.
    y1 : np.meshgrid
        Grid of y components for the contour plot.
    x2 : np.mesgrid
        Grid of x components for the colormesh plot.
    y2 : np.meshgrid
        Grid of y components for the colormesh plot.
    V : np.array
        Matrix of the final potential values.
    lvl : list or float, optional
        Number of equipotential levels or list of specific levels. The default
        is 10.

    Returns
    -------
    None.

    """
    plt.pcolormesh(x2, y2, V, cmap='plasma')
    plt.colorbar(label=r'$\Phi \ (\si{\volt})$')
    plt.contour(x1, y1, V, levels=lvl,
                colors='k', linewidths=1, linestyles='dotted')

    for ch in charges:
        ch.plot()

    plt.xlabel(r'$x \ (\si{\meter})$')
    plt.ylabel(r'$y \ (\si{\meter})$')
    return None
