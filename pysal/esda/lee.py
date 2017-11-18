"""
Lee's L Bivariate Spatial Association Measure
Addresses two problems with the Bivariate Moran for measuring spatial associations:
    1. Lee's L is symmetric. L(var1, var2) == L(var2, var1)
    2. Lee's L factors in _both_ variables' spatial autocorrelations.

As defined in: 
Lee, SI. J Geograph Syst (2001) 3: 369. https://doi.org/10.1007/s101090100064
"""
__author__ = "Nicholas W. Tomasino <nick.w.tomasino@gmail.com>"

from ..weights.spatial_lag import lag_spatial as slag
from .smoothing import assuncao_rate
from .tabular import _univariate_handler, _bivariate_handler
import scipy.stats as stats
import numpy as np

__all__ = ["Lee", "Lee_Local"]
#TODO Rates
#__all__ = ["L_Stat", "L_Stat_Local", "L_Stat_Rate", "L_Stat_Local_Rate"]

PERMUTATIONS = 999


class Lee(object):
    """
    Lee's Bivariate Spatial Association Measure

    L = Sum_i( slag(zx_i) * slag(zy_i) )

    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        y-axis variable
        wy will be on y axis TODO delete me
    w : W
        weight instance assumed to be aligned with y
    transformation  : {'R', 'B', 'D', 'U', 'V'}
                      weights transformation, default is row-standardized "r".
                      Other options include
                      "B": binary,
                      "D": doubly-standardized,
                      "U": untransformed (general weights),
                      "V": variance-stabilizing.
    permutations    : int
                      number of random permutations for calculation of pseudo
                      p_values

    Attributes
    ----------
    zx            : array
                    original x variable standardized by mean and std
    zy            : array
                    original y variable standardized by mean and std
    w             : W
                    original w object
    permutation   : int
                    number of permutations
    L             : float
                    value of Lee's L
    sim           : array
                    (if permutations>0)
                    vector of L values for permuted samples
    p_sim         : float
                    (if permutations>0)
                    p-value based on permutations (one-sided)
                    null: spatial randomness
                    alternative: the observed L is extreme
                    it is either extremely high or extremely low
    EL_sim        : array
                    (if permutations>0)
                    average value of L from permutations
    VL_sim        : array
                    (if permutations>0)
                    variance of L from permutations
    seL_sim       : array
                    (if permutations>0)
                    standard deviation of L under permutations.
    z_sim         : array
                    (if permutations>0)
                    standardized L based on permutations
    p_z_sim       : float
                    (if permutations>0)
                    p-value based on standard normal approximation from
                    permutations

    Notes
    -----

    Inference is only based on permutations as analytical results are none too
    reliable.

    Examples
    --------
    >>> import pysal
    >>> import numpy as np

    Set random number generator seed so we can replicate the example

    >>> np.random.seed(10)

    Open the sudden infant death dbf file and read in rates for 74 and 79
    converting each to a numpy array

    >>> f = pysal.open(pysal.examples.get_path("sids2.dbf"))
    >>> SIDR74 = np.array(f.by_col['SIDR74'])
    >>> SIDR79 = np.array(f.by_col['SIDR79'])

    Read a GAL file and construct our spatial weights object

    >>> w = pysal.open(pysal.examples.get_path("sids2.gal")).read()

    Create an instance of Lee

    >>> lee = pysal.Lee(SIDR79,  SIDR74,  w)

    >>> print lee.L
    0.0880162449211

    >>> print lee.p_z_sim
    0.00815369861243


    """
    def __init__(self, x, y, w, transformation="r", permutations=PERMUTATIONS):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        zy = (y - y.mean()) / y.std(ddof=1)
        zx = (x - x.mean()) / x.std(ddof=1)
        self.zx = zx
        self.zy = zy
        n = x.shape[0]
        self.den = n - 1.  # zx'zx = zy'zy = n-1  #TODO double-check den
        w.transform = transformation
        self.w = w
        self.L = self.__calc(zx, zy)
        if permutations:
            nrp = np.random.permutation
            sim = [self.__calc(nrp(zx), nrp(zy)) for i in xrange(permutations)]
            self.sim = sim = np.array(sim)
            above = sim >= self.L
            larger = above.sum()
            if (permutations - larger) < larger:
                larger = permutations - larger
            self.p_sim = (larger + 1.) / (permutations + 1.)
            self.EL_sim = sim.sum() / permutations
            self.seL_sim = np.array(sim).std()
            self.VL_sim = self.seL_sim ** 2
            self.z_sim = (self.L - self.EL_sim) / self.seL_sim
            if self.z_sim > 0:
                self.p_z_sim = 1 - stats.norm.cdf(self.z_sim)
            else:
                self.p_z_sim = stats.norm.cdf(self.z_sim)

    def __calc(self, zx, zy):
        zlx = slag(self.w, zx)
        zly = slag(self.w, zy)
        self.num = (zlx * zly).sum()
        return self.num / self.den

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.L
    
    @classmethod
    def by_col(cls, df, x, y=None, w=None, inplace=False, pvalue='sim', outvals=None, **stat_kws):
        """ 
        Function to compute a Moran_BV statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        x           :   list of strings
                        column name or list of column names to use as X values to compute
                        the bivariate statistic. If no Y is provided, pairwise comparisons
                        among these variates are used instead. 
        y           :   list of strings
                        column name or list of column names to use as Y values to compute
                        the bivariate statistic. if no Y is provided, pariwise comparisons
                        among the X variates are used instead. 
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_moran_local'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Moran_BV statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the 
                        Moran_BV statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Moran_BV statistic.


        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        See Also
        ---------
        For further documentation, refer to the Moran_BV class in pysal.esda
        """
        return _bivariate_handler(df, x, y=y, w=w, inplace=inplace, 
                                  pvalue = pvalue, outvals = outvals, 
                                  swapname=cls.__name__.lower(), stat=cls,**stat_kws)


class Lee_Local(object):
    """Local Lee Statistics
    L_i = n*slag(zx_i)*slag(zy_i)


    Parameters
    ----------
    x : array
        x-axis variable
    y : array
        (n,1), wy will be on y axis
    w : W
        weight instance assumed to be aligned with y
    transformation : {'R', 'B', 'D', 'U', 'V'}
                     weights transformation,  default is row-standardized "r".
                     Other options include
                     "B": binary,
                     "D": doubly-standardized,
                     "U": untransformed (general weights),
                     "V": variance-stabilizing.
    permutations   : int
                     number of random permutations for calculation of pseudo
                     p_values
    geoda_quads    : boolean
                     (default=False)
                     If True use GeoDa scheme: HH=1, LL=2, LH=3, HL=4
                     If False use PySAL Scheme: HH=1, LH=2, LL=3, HL=4

    Attributes
    ----------

    zx           : array
                   original x variable standardized by mean and std
    zy           : array
                   original y variable standardized by mean and std
    w            : W
                   original w object
    permutations : int
                   number of random permutations for calculation of pseudo
                   p_values
    Ls           : float
                   value of Lee's L
    q            : array
                   (if permutations>0)
                   values indicate quandrant location 1 HH,  2 LH,  3 LL,  4 HL
    sim          : array
                   (if permutations>0)
                   vector of L values for permuted samples
    p_sim        : array
                   (if permutations>0)
                   p-value based on permutations (one-sided)
                   null: spatial randomness
                   alternative: the observed Li is further away or extreme
                   from the median of simulated values. It is either extremely
                   high or extremely low in the distribution of simulated Ls.
    EL_sim       : array
                   (if permutations>0)
                   average values of local Ls from permutations
    VL_sim       : array
                   (if permutations>0)
                   variance of Ls from permutations
    seL_sim      : array
                   (if permutations>0)
                   standard deviations of Ls under permutations.
    z_sim        : arrray
                   (if permutations>0)
                   standardized Ls based on permutations
    p_z_sim      : array
                   (if permutations>0)
                   p-values based on standard normal approximation from
                   permutations (one-sided)
                   for two-sided tests, these values should be multiplied by 2


    Examples
    --------
    >>> import pysal as ps
    >>> import numpy as np
    >>> np.random.seed(10)
    >>> w = ps.open(ps.examples.get_path("sids2.gal")).read()
    >>> f = ps.open(ps.examples.get_path("sids2.dbf"))
    >>> x = np.array(f.by_col['SIDR79'])
    >>> y = np.array(f.by_col['SIDR74'])
    >>> lee = ps.Lee_Local(x, y, w, transformation = "r", \
                               permutations = 99)
    >>> lee.q[:10]
    array([3, 4, 3, 4, 2, 1, 4, 4, 2, 4])
    >>> lee.L[0]
    0.0000000000000000000
    >>> lee.p_z_sim[0]
    0.0017240031348827456

    Note random components result is slightly different values across
    architectures so the results have been removed from doctests and will be
    moved into unittests that are conditional on architectures
    """
    def __init__(self, x, y, w, transformation="r", permutations=PERMUTATIONS,
                 geoda_quads=False):
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        self.y = y
        n = len(y)
        self.n = n
        self.n_1 = n - 1 #TODO why n-1 instead of n? Lee's paper has n
        zx = x - x.mean()
        zy = y - y.mean()
        # setting for floating point noise
        orig_settings = np.seterr()
        np.seterr(all="ignore")
        sx = x.std()
        zx /= sx
        sy = y.std()
        zy /= sy
        np.seterr(**orig_settings)
        self.zx = zx
        self.zy = zy
        w.transform = transformation
        self.w = w
        self.permutations = permutations
        #self.den = (zx * zx).sum() #TODO unsure if needed
        self.den = (zx * zy).sum() #TODO unsure if needed
        self.Ls = self.calc(self.w, self.zx, self.zy)
        self.geoda_quads = geoda_quads
        quads = [1, 2, 3, 4]
        if geoda_quads:
            quads = [1, 3, 2, 4]
        self.quads = quads
        self.__quads()
        if permutations:
            self.__crand()
            sim = np.transpose(self.rlisas)
            above = sim >= self.Ls
            larger = above.sum(0)
            low_extreme = (self.permutations - larger) < larger
            larger[low_extreme] = self.permutations - larger[low_extreme]
            self.p_sim = (larger + 1.0) / (permutations + 1.0)
            self.sim = sim
            self.EL_sim = sim.mean(axis=0)
            self.seL_sim = sim.std(axis=0)
            self.VL_sim = self.seL_sim * self.seL_sim
            self.z_sim = (self.Ls - self.EL_sim) / self.seL_sim
            self.p_z_sim = 1 - stats.norm.cdf(np.abs(self.z_sim))

    def calc(self, w, zx, zy):
        zlx = slag(w, zx)
        zly = slag(w, zy)
        return zlx * zly
        #convert back from z-vals?
        #return self.n_1 * zlx * zly #nope
        #return self.n_1 * zlx * zly / self.den
        #return zlx * zly / self.den

    def __crand(self):
        """
        conditional randomization

        for observation i with ni neighbors,  the candidate set cannot include
        i (we don't want i being a neighbor of i). we have to sample without
        replacement from a set of ids that doesn't include i. numpy doesn't
        directly support sampling wo replacement and it is expensive to
        implement this. instead we omit i from the original ids,  permute the
        ids and take the first ni elements of the permuted ids as the
        neighbors to i in each randomization.

        """
        lisas = np.zeros((self.n, self.permutations))
        n_1 = self.n - 1
        prange = range(self.permutations)
        k = self.w.max_neighbors + 1
        nn = self.n - 1
        rids = np.array([np.random.permutation(nn)[0:k] for i in prange])
        ids = np.arange(self.w.n)
        ido = self.w.id_order
        w = [self.w.weights[ido[i]] for i in ids]
        wc = [self.w.cardinalities[ido[i]] for i in ids]

        zx = self.zx
        zy = self.zy
        for i in xrange(self.w.n):
            idsi = ids[ids != i]
            np.random.shuffle(idsi)
            tmp = zy[idsi[rids[:, 0:wc[i]]]]
            lisas[i] = zx[i] * (w[i] * tmp).sum(1)
        self.rlisas = (n_1 / self.den) * lisas

    def __quads(self):
        zl = slag(self.w, self.zy)
        zp = self.zx > 0
        lp = zl > 0
        pp = zp * lp
        np = (1 - zp) * lp
        nn = (1 - zp) * (1 - lp)
        pn = zp * (1 - lp)
        self.q = self.quads[0] * pp + self.quads[1] * np + self.quads[2] * nn \
            + self.quads[3] * pn

    @property
    def _statistic(self):
        """More consistent hidden attribute to access ESDA statistics"""
        return self.Ls

    @classmethod
    def by_col(cls, df, x, y=None, w=None, inplace=False, pvalue='sim', outvals=None, **stat_kws):
        """ 
        Function to compute a Lee_Local statistic on a dataframe

        Arguments
        ---------
        df          :   pandas.DataFrame
                        a pandas dataframe with a geometry column
        X           :   list of strings
                        column name or list of column names to use as X values to compute
                        the bivariate statistic. If no Y is provided, pairwise comparisons
                        among these variates are used instead. 
        Y           :   list of strings
                        column name or list of column names to use as Y values to compute
                        the bivariate statistic. if no Y is provided, pariwise comparisons
                        among the X variates are used instead. 
        w           :   pysal weights object
                        a weights object aligned with the dataframe. If not provided, this
                        is searched for in the dataframe's metadata
        inplace     :   bool
                        a boolean denoting whether to operate on the dataframe inplace or to
                        return a series contaning the results of the computation. If
                        operating inplace, the derived columns will be named
                        'column_lee_local'
        pvalue      :   string
                        a string denoting which pvalue should be returned. Refer to the
                        the Lee_Local statistic's documentation for available p-values
        outvals     :   list of strings
                        list of arbitrary attributes to return as columns from the 
                        Lee_Local statistic
        **stat_kws  :   keyword arguments
                        options to pass to the underlying statistic. For this, see the
                        documentation for the Lee_Local statistic.


        Returns
        --------
        If inplace, None, and operation is conducted on dataframe in memory. Otherwise,
        returns a copy of the dataframe with the relevant columns attached.

        See Also
        ---------
        For further documentation, refer to the Lee_Local class in pysal.esda
        """
        return _bivariate_handler(df, x, y=y, w=w, inplace=inplace, 
                                  pvalue = pvalue, outvals = outvals, 
                                  swapname=cls.__name__.lower(), stat=cls,**stat_kws)
