# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Robert Weld
# Last Modified: 14/02/22

import world
import random
import utils
import time
import config
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
from utils import Directions
import mdptoolbox

class Tallon():

    def __init__(self, arena):
        """
        initialises variables that need to stored
        and accessed through running of class

        Parameters:
        arena (world): copy of the world  so that Tallon can
                    query the state of the world

         """

        # What moves are possible.
        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        
        #dimensions of arena
        self.gameWorld = arena
        arenaWidth = arena.maxX+1
        arenaHight = arena.maxY+1
        self.arenaSize = arenaWidth * arenaHight

        #chance for move to be successful
        moveChance = config.directionProbability
        misstepChance = 1-moveChance

        #init transition model/probability array
        self.probArray = self.generateProbArray(arenaWidth,moveChance,misstepChance)

        #target for Tallon if there are no bonuses
        self.noBonusX = 0
        self.noBunusY = 0
    
    def generateProbArray(self,arenaWidth,moveChance,misstepChance):
        """create's transition model/probability array for 

        Parameters:
        arenaWidth (int): width of arena to tell length of rows
        moveChance (float): chance of succefully moving as planned
        misstepChance (float): chance moving off to the side of planned move

        Returns:
        ndarray: Returning probArray

        """
        
        probArray = np.zeros((len(self.moves),self.arenaSize,self.arenaSize))
        
        north,south,east,west = range(len(self.moves))

        for i in range(self.arenaSize):
            #eastern edge of arena
            if (i % arenaWidth) == 0:
                probArray[west][i][i] = moveChance
                probArray[east][i][i+1] = moveChance
                probArray[north][i][i+1] = misstepChance
                probArray[south][i][i+1] = misstepChance
            #western edge of arena
            elif ((i + 1) % arenaWidth) == 0:
                probArray[east][i][i] = moveChance
                probArray[west][i][i-1] = moveChance
                probArray[north][i][i-1] = misstepChance
                probArray[south][i][i-1] = misstepChance
            #middle columns
            else:
                probArray[east][i][i+1] = moveChance
                probArray[west][i][i-1] = moveChance
                probArray[north][i][i+1] = misstepChance*0.5
                probArray[north][i][i-1] = misstepChance*0.5
                probArray[south][i][i+1] = misstepChance*0.5
                probArray[south][i][i-1] = misstepChance*0.5
            #northern edge of arena
            if i < arenaWidth:
                probArray[north][i][i] = moveChance
                probArray[south][i][i+arenaWidth] = moveChance
                probArray[east][i][i+arenaWidth] = misstepChance
                probArray[west][i][i+arenaWidth] = misstepChance
            #southern edge of arena
            elif i >= (self.arenaSize-arenaWidth):
                probArray[south][i][i] = moveChance
                probArray[north][i][i-arenaWidth] = moveChance
                probArray[east][i][i-arenaWidth] = misstepChance
                probArray[west][i][i-arenaWidth] = misstepChance
            #middle rows
            else:
                probArray[north][i][(i-arenaWidth)] = moveChance
                probArray[south][i][(i+arenaWidth)] = moveChance
                probArray[east][i][i+arenaWidth] = misstepChance*0.5
                probArray[east][i][i-arenaWidth] = misstepChance*0.5
                probArray[west][i][i+arenaWidth] = misstepChance*0.5
                probArray[west][i][i-arenaWidth] = misstepChance*0.5

        print(probArray)
        return probArray
    
    def makeMove(self):
        """calculates the direction and returns direction with 
        highest utility for Tallon's current location to avoid
        meanies, pits and go to bonuses

        Returns:
        enum: Returning direction to move tallon

        """
        locations = self.getLocations()
        north,south,east,west = range(len(self.moves))
        rewards = np.zeros(((self.arenaSize),len(self.moves)))
        result = locations.flatten('F')

        for i in range(len(result)):
            rewards[i][north] = result[i]
            rewards[i][south] = result[i]
            rewards[i][east] = result[i]
            rewards[i][west] = result[i]

        mdptoolbox.util.check(self.probArray, rewards)
        vi2 = mdptoolbox.mdp.ValueIteration(self.probArray, rewards, 0.9)
        vi2.run()

        myPosition = self.gameWorld.getTallonLocation()
        whatPolicy = myPosition.x + (myPosition.y*(self.gameWorld.maxY+1))
        movement = vi2.policy[whatPolicy]
        
        print((myPosition.x,myPosition.y))
        print("Policy: {0}".format(whatPolicy))

        if movement == north:
            print("Tallon goes north")
            return Directions.NORTH
        if movement == south:
            print("Tallon goes south")
            return Directions.SOUTH
        if movement == east:
            print("Tallon goes east")
            return Directions.EAST
        if movement == west:
            print("Tallon goes west")
            return Directions.WEST
        
    def getLocations(self):
        """gets locations of pits, meanies and Bonuses(or navigation goals if no bonuses in sight)
        and assigns -1 for pits/meanies and 1 for bonuses/navigation goals for Tallon

        Returns:
        ndarray: Returning 2d array rewards corespinding as negatives to avoid and postives as goals

        """
        allPits = self.gameWorld.getPitsLocation()
        allBonuses = self.gameWorld.getBonusLocation()
        allMeanies = self.gameWorld.getMeanieLocation()
        locations = np.zeros((self.gameWorld.maxX+1,self.gameWorld.maxY+1))

        myPosition = self.gameWorld.getTallonLocation()
        locations[myPosition.x][myPosition.y] = -0.4

        #move around to each corner of the map if no bonuses
        if len(allBonuses) == 0:
            if myPosition.x < (self.gameWorld.maxX*0.25) and myPosition.y > (self.gameWorld.maxY*0.75):
                self.noBonusX = 0
                self.noBunusY = 0
            if myPosition.x > (self.gameWorld.maxX*0.75) and myPosition.y > (self.gameWorld.maxY*0.75):
                self.noBonusX = 0
                self.noBunusY = self.gameWorld.maxY 
            if myPosition.x < (self.gameWorld.maxX*0.25) and myPosition.y < (self.gameWorld.maxY*0.25):
                self.noBonusX = self.gameWorld.maxX
                self.noBunusY = 0
            if myPosition.x > (self.gameWorld.maxX*0.75) and myPosition.y < (self.gameWorld.maxY*0.25):
                self.noBonusX = self.gameWorld.maxX
                self.noBunusY = self.gameWorld.maxY

            locations[self.noBonusX][self.noBunusY] = 1
            print("Goal: ({0},{1})".format(self.noBonusX,self.noBunusY))

        for b in allBonuses:
            locations[b.x][b.y] = 1

        for p in allPits:
            locations[p.x][p.y] = -1

        for m in allMeanies:
            locations[m.x][m.y] = -1

        return locations
