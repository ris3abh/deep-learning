from abc import ABC, abstractmethod 

class Layer (ABC) :
    def init (self): 
        self . prevIn = [] 
        self. prevOut=[]

    def setPrevIn(self ,dataIn): 
        self . prevIn = dataIn

    def setPrevOut( self , out ): 
        self . prevOut = out

    def getPrevIn( self ):
        return self . prevIn

    def getPrevOut( self ):
        return self . prevOut

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward( self , gradIn ):
        pass