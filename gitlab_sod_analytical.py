"""
Source: https://gitlab.com/fantaz/simple_shock_tube_calculator

This module calculates analytical solutions for a Sod shock tube problem. 
The project was created by user `fantaz` on GitLab.
"""
import numpy as np
import scipy
import scipy.optimize


def solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.), geometry=(-0.5, 0.5, 0), t=0.2, **kwargs):
    """
    Solves the Sod shock tube problem (i.e. riemann problem) of discontinuity across an interface.

    :rtype : tuple
    :param left_state: tuple (pl, rhol, ul)
    :param right_state: tuple (pr, rhor, ur)
    :param geometry: tuple (xl, xr, xi): xl - left boundary, xr - right boundary, xi - initial discontinuity
    :param t: time for which the states have to be calculated
    :param gamma: ideal gas constant, default is air: 1.4
    :param npts: number of points for array of pressure, density and velocity
    :param dustFrac: dust to gas fraction, should be >=
    :return: tuple of: dicts of positions,
    constant pressure, density and velocity states in distinct regions,
    arrays of pressure, density and velocity in domain bounded by xl, xr
    """

    if 'npts' in kwargs:
        npts = kwargs['npts']
    else:
        npts = 500

    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
    else:
        gamma = 1.4

    if 'dustFrac' in kwargs:
        # dustFrac = np.min(np.max(kwargs['dustFrac'], 0), 1)
        dustFrac = kwargs['dustFrac']
        if dustFrac<0 or dustFrac >= 1:
            print('Invalid dust fraction value: {}. Should be >=0 and <1. Set to default: 0'.format(dustFrac))
            dustFrac = 0
    else:
        dustFrac = 0

    calculator = Calculator(left_state=left_state, right_state=right_state, geometry=geometry, t=t,
                            gamma=gamma, npts=npts, dustFrac=dustFrac)

    return calculator.solve()


class Calculator:
    """
    Class that does the actual work computing the Sod shock tube problem
    """

    def __init__(self, left_state, right_state, geometry, t, **kwargs):
        """
        Ctor
        :param left_state: tuple (pl, rhol, ul)
        :param right_state: tuple (pr, rhor, ur)
        :param geometry: tuple (xl, xr, xi): xl - left boundary, xr - right boundary, xi - initial discontinuity
        :param t: time for which the states have to be calculated
        :param gamma: ideal gas constant, default is air: 1.4
        :param npts: number of points for array of pressure, density and velocity
        :param dustFrac: dust fraction, as defined in Tricco, Price and Laibe, 2017,
        Is the dust-to-gas ratio constant in molecular clouds, Eqs. 5-6
        :param kwargs:
        """
        self.pl, self.rhol, self.ul = left_state
        self.pr, self.rhor, self.ur = right_state
        self.xl, self.xr, self.xi = geometry
        self.t = t

        if 'npts' in kwargs:
            self.npts = kwargs['npts']
        else:
            self.npts = 500

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 1.4

        if 'dustFrac' in kwargs:
            self.dustFrac = kwargs['dustFrac']
        else:
            self.dustFrac = 0

        # basic checking
        if self.xl >= self.xr:
            print('xl has to be less than xr!')
            exit()
        if self.xi >= self.xr or self.xi <= self.xl:
            print('xi has in between xl and xr!')
            exit()

        # calculate regions
        self.region1, self.region3, self.region4, self.region5, self.w = \
            self.calculate_regions()

    def solve(self):
        """
        Actually solves the sod shock tube problem
        :return:
        """
        regions = self.region_states()

        # calculate positions
        x_positions = self.calc_positions()

        pos_description = ('Head of Rarefaction', 'Foot of Rarefaction',
                           'Contact Discontinuity', 'Shock')
        positions = dict(zip(pos_description, x_positions))

        # create arrays
        x, p, rho, u = self.create_arrays(x_positions)

        val_names = ('x', 'p', 'rho', 'u')
        val_dict = dict(zip(val_names, (x, p, rho, u)))

        return positions, regions, val_dict

    def sound_speed(self, p, rho):
        """
        Calculate speed of sound according to

            .. math::
                c = \sqrt{\gamma \frac{p}{\rho} (1-\theta)}
        where :math:`p` is pressure, :math:`\rho` is density, :math:`\gamma` is heat capacity ratio
        and :math:`\theta` is dust fraction, according to Tricco, Price and Laibe, 2017

        :rtype : float
        :return: returns the speed of sound
        """
        return np.sqrt(self.gamma * (1-self.dustFrac) * p / rho)

    def shock_tube_function(self, p4, p1, p5, rho1, rho5):
        """
        Shock tube equation
        """
        z = (p4 / p5 - 1.)
        c1 = self.sound_speed(p1, rho1)
        c5 = self.sound_speed(p5, rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        g2 = 2. * self.gamma

        fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1. + gp1 / g2 * z)
        fact = (1. - fact) ** (g2 / gm1)

        return p1 * fact - p4

    def calculate_regions(self):
        """
        Compute regions
        :rtype : tuple
        :return: returns p, rho and u for regions 1,3,4,5 as well as the shock speed
        """
        # if pl > pr...
        rho1 = self.rhol
        p1 = self.pl
        u1 = self.ul
        rho5 = self.rhor
        p5 = self.pr
        u5 = self.ur

        # unless...
        if self.pl < self.pr:
            rho1 = self.rhor
            p1 = self.pr
            u1 = self.ur
            rho5 = self.rhol
            p5 = self.pl
            u5 = self.ul

        # solve for post-shock pressure
        # just in case the shock_tube_function gets a complex number
        num_of_guesses = 100
        for pguess in np.linspace(self.pr, self.pl, num_of_guesses):
            res = scipy.optimize.fsolve(self.shock_tube_function, pguess, args=(p1, p5, rho1, rho5), full_output=True)
            p4, infodict, ier, mesg = res
            if ier == 1:
                break
        if not ier == 1:
            raise Exception("Analytical Sod solution unsuccessful!")

        if type(p4) is np.ndarray:
            p4 = p4[0]

        # compute post-shock density and velocity
        z = (p4 / p5 - 1.)
        c5 = self.sound_speed(p5, rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        gmfac1 = 0.5 * gm1 / self.gamma
        gmfac2 = 0.5 * gp1 / self.gamma

        fact = np.sqrt(1. + gmfac2 * z)

        u4 = c5 * z / (self.gamma * fact)
        rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z)

        # shock speed
        w = c5 * fact

        # compute values at foot of rarefaction
        p3 = p4
        u3 = u4
        rho3 = rho1 * (p3 / p1) ** (1. / self.gamma)
        return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w

    def region_states(self):
        """
        :return: dictionary (region no.: p, rho, u), except for rarefaction region
        where the value is a string, obviously
        """
        if self.pl > self.pr:
            return {'Region 1': self.region1,
                    'Region 2': 'RAREFACTION',
                    'Region 3': self.region3,
                    'Region 4': self.region4,
                    'Region 5': self.region5}
        else:
            return {'Region 1': self.region5,
                    'Region 2': self.region4,
                    'Region 3': self.region3,
                    'Region 4': 'RAREFACTION',
                    'Region 5': self.region1}

    def calc_positions(self):
        """
        :return: tuple of positions in the following order ->
                Head of Rarefaction: xhd,  Foot of Rarefaction: xft,
                Contact Discontinuity: xcd, Shock: xsh
        """
        p1, rho1 = self.region1[:2]  # don't need velocity
        p3, rho3, u3 = self.region3[:]
        c1 = self.sound_speed(p1, rho1)
        c3 = self.sound_speed(p3, rho3)
        if self.pl > self.pr:
            xsh = self.xi + self.w * self.t
            xcd = self.xi + u3 * self.t
            xft = self.xi + (u3 - c3) * self.t
            xhd = self.xi - c1 * self.t
        else:
            # pr > pl
            xsh = self.xi - self.w * self.t
            xcd = self.xi - u3 * self.t
            xft = self.xi - (u3 - c3) * self.t
            xhd = self.xi + c1 * self.t

        return xhd, xft, xcd, xsh

    def create_arrays(self, positions):
        """
        :return: tuple of x, p, rho and u values across the domain of interest
        """
        xhd, xft, xcd, xsh = positions
        p1, rho1, u1 = self.region1
        p3, rho3, u3 = self.region3
        p4, rho4, u4 = self.region4
        p5, rho5, u5 = self.region5
        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.

        x_arr = np.linspace(self.xl, self.xr, self.npts)
        rho = np.zeros(self.npts, dtype=float)
        p = np.zeros(self.npts, dtype=float)
        u = np.zeros(self.npts, dtype=float)
        c1 = self.sound_speed(p1, rho1)

        if self.pl > self.pr:
            for i, x in enumerate(x_arr):
                if x < xhd:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                elif x < xft:
                    u[i] = 2. / gp1 * (c1 + (x - self.xi) / self.t)
                    fact = 1. - 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                elif x < xcd:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                elif x < xsh:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                else:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = u5
        else:
            for i, x in enumerate(x_arr):
                if x < xsh:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = -u1
                elif x < xcd:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = -u4
                elif x < xft:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = -u3
                elif x < xhd:
                    u[i] = -2. / gp1 * (c1 + (self.xi - x) / self.t)
                    fact = 1. + 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                else:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = -u1

        return x_arr, p, rho, u
