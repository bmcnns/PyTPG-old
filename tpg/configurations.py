class DefaultConfiguration:
    def __init__(self):
        self.teamPopSize = 360
        self.rootBasedPop = True
        self.sharedMemory = True
        self.gap=0.5
        self.uniqueProgThresh=0
        self.initMaxTeamSize=5
        self.initMaxProgSize=128
        
        # unused ??
        #actionProgSize=64

        self.registerSize=8
        self.pDelLrn=0.7
        self.pAddLrn=0.7
        self.pMutLrn=0.3
        self.pMutProg=0.66
        self.pMutAct=0.33
        self.pActAtom=0.5
        self.pDelInst=0.5
        self.pAddInst=0.5
        self.pSwpInst=1.0
        self.pMutInst=1.0

        # check if this is being used outside trainer.py
        self.pSwapMultiAct=0.66
        
        # same as above
        self.pChangeMultiAct=0.40

        self.doElites=True
        self.sourceRange=30720

        self.memorySize = 400


    

