'''
Licensing Information: Please do not distribute or publish solutions to this
project. You are free to use and extend Driverless Car for educational
purposes. The Driverless Car project was developed at Stanford, primarily by
Chris Piech (piech@cs.stanford.edu). It was inspired by the Pacman projects.
'''
import collections
import math
import random
import util
from engine.const import Const
from util import Belief


class ExactInference:
    """ 
    Maintain and update a belief distribution over the probability of a car
    being in a tile using exact updates (correct, but slow times).
    """

    def __init__(self, numRows: int, numCols: int):
        """
        Constructor that initializes an ExactInference object which has
        numRows x numCols number of tiles.
        """
        self.skipElapse = False  ### ONLY USED BY GRADER.PY in case problem 2 has not been completed
        # util.Belief is a class (constructor) that represents the belief for a single
        # inference state of a single car (see util.py).
        self.belief = util.Belief(numRows, numCols)
        self.transProb = util.loadTransProb()

    ##################################################################################
    # Problem 1:
    # Function: Observe (update the probabilities based on an observation)
    # -----------------
    # Takes |self.belief| -- an object of class Belief, defined in util.py --
    # and updates it in place based on the distance observation $d_t$ and
    # your position $a_t$.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    # - Don't forget to normalize self.belief after you update its probabilities!
    ##################################################################################

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        # Bayes' rule for a hidden location C and observed distance D_t = d_t:
        #   P(C = c | d_1..d_t)  ∝  P(C = c | d_1..d_{t-1}) * p(d_t | c),
        # where the emission p(d_t | c) is a Gaussian density centred on the true distance
        # between our car (agentX, agentY) and the centre of tile c, with std Const.SONAR_STD.
        for row in range(self.belief.getNumRows()):
            for col in range(self.belief.getNumCols()):
                true_dist = math.hypot(agentX - util.colToX(col), agentY - util.rowToY(row))
                emission = util.pdf(true_dist, Const.SONAR_STD, observedDist)  # pdf(mean, std, value)
                self.belief.setProb(row, col, self.belief.getProb(row, col) * emission)
        self.belief.normalize()
        # END_YOUR_CODE

    ##################################################################################
    # Problem 2:
    # Function: Elapse Time (propose a new belief distribution based on a learned transition model)
    # ---------------------
    # Takes |self.belief| and updates it based on the passing of one time step.
    # Notes:
    # - Tile coordinates are represented as (row, col) tuples
    # - Use the transition probabilities in self.transProb, which is a dictionary
    #   containing all the ((oldTile, newTile), transProb) key-val pairs that you
    #   must consider
    # - If there are ((oldTile, newTile), transProb) pairs not in self.transProb,
    #   they are assumed to have zero probability, and you can safely ignore them.
    # - Use the addProb (or setProb) and getProb methods of the Belief class to modify
    #   and access the probabilities associated with a belief.  (See util.py.)
    # - Be careful that you are using only the CURRENT self.belief distribution to compute
    #   updated beliefs.  Don't incrementally update self.belief and use the updated value
    #   for one grid square to compute the update for another square.
    # - Don't forget to normalize self.belief after all probabilities have been updated!
    #   (so that the sum of probabilities is exactly 1 as otherwise adding/multiplying
    #    small floating point numbers can lead to sum being close to but not equal to 1)
    ##################################################################################
    def elapseTime(self) -> None:
        if self.skipElapse: ### ONLY FOR THE GRADER TO USE IN Problem 1
            return
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        # Forward step of the HMM:
        #   P(C_{t+1} = c' | d_1..d_t) ∝ sum_c P(C_t = c | d_1..d_t) * p(c' | c).
        # Build the new belief in a separate grid (all zeros) so that every term uses only the
        # *current* belief, then iterate over the sparse transition table (pairs not listed
        # have probability 0).
        new_belief = util.Belief(self.belief.getNumRows(), self.belief.getNumCols(), 0.0)
        for (old_tile, new_tile), trans_prob in self.transProb.items():
            new_belief.addProb(new_tile[0], new_tile[1],
                               self.belief.getProb(old_tile[0], old_tile[1]) * trans_prob)
        new_belief.normalize()
        self.belief = new_belief
        # END_YOUR_CODE

    def getBelief(self) -> Belief:
        """
        Returns your belief of the probability that the car is in each tile. Your
        belief probabilities should sum to 1.
        """
        return self.belief


class ExactInferenceWithSensorDeception(ExactInference):
    """
    Same as ExactInference except with sensor deception attack represented in the
    observation function.
    """

    def __init__(self, numRows: int, numCols: int, skewness: float = 0.5):
        """
        Constructor that initializes an ExactInference object which has
        numRows x numCols number of tiles, as well as a skewness factor
        used to calculate the skewed observed distance distribution.
        """
        super().__init__(numRows, numCols)
        self.skewness = skewness

    ##################################################################################
    # Problem 4:
    # Function: Observe with sensor deception (update the probabilities based on an observation)
    # -----------------
    # Apply the adjustment to observed distance based on the transformation
    # D_t_' = 1/(1+skewness**2) * D_t + sqrt(2 * (1/(1+skewness**2))) then copy
    # your previous observe() implementation from ExactInference() to update the probabilities.
    # You could also call the parent class' observe(x, y, dist) method in place of copying
    # the implementation, but either approach is acceptable.
    # Note that the skewness parameter is set in the constructor.
    #
    # - agentX: x location of your car (not the one you are tracking)
    # - agentY: y location of your car (not the one you are tracking)
    # - observedDist: true distance plus a mean-zero Gaussian with standard
    #                 deviation Const.SONAR_STD
    #
    # Notes:
    # - Convert row and col indices into locations using util.rowToY and util.colToX.
    # - util.pdf: computes the probability density function for a Gaussian
    # - Although the gaussian pdf is symmetric with respect to the mean and value,
    #   you should pass arguments to util.pdf in the correct order
    # - Don't forget to normalize self.belief after you update its probabilities!
    ##################################################################################

    def observe(self, agentX: int, agentY: int, observedDist: float) -> None:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        # The attacker skews every reading: D'_t = D_t / (1 + s^2) + sqrt(2 / (1 + s^2)).
        # Apply that transformation, then run the ordinary emission update on the skewed value.
        shrink = 1.0 / (1.0 + self.skewness ** 2)
        skewed_dist = shrink * observedDist + math.sqrt(2.0 * shrink)
        super().observe(agentX, agentY, skewed_dist)
        # END_YOUR_CODE

    def elapseTime(self) -> None:
        super().elapseTime()

    def getBelief(self) -> Belief:
        return super().getBelief()
