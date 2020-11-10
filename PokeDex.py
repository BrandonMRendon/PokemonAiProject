import pandas as pd

class Pokemon:
  def __init__(self,name,type,move1,move2,move3,move4,move1type,move2type,move3type,move4type):
    self.name = name
    self.type = type
    self.move1 = move1
    self.move2 = move2
    self.move3 = move3
    self.move4 = move4
    self.move1type = move1type
    self.move2type = move2type
    self.move3type = move3type
    self.move4type = move4type
def printPoke(Pokemon):
    print(Pokemon.name + " ~ " + Pokemon.type + "\n")
    print(Pokemon.move1 + " ~ " + Pokemon.move1type + "\n")
    print(Pokemon.move2 + " ~ " + Pokemon.move2type + "\n")
    print(Pokemon.move3 + " ~ " + Pokemon.move3type + "\n")
    print(Pokemon.move4 + " ~ " + Pokemon.move4type + "\n\n\n")

def makeDex():
    PokeDex = []
    df = pd.read_csv('PokeDex - Sheet1.csv')
    name = "Charizard"
    index = "Name", "Types", "Move1", "Move2", "Move3", "Move4", "Move1Type", "Move2Type", "Move3Type", "Move4Type"
    print(df)
    print(df[index[0]][0])
    count1=0
    while (count1 < 9):
        count2 = 0
        data = []
        while (count2 < 10):
            data.append(df[index[count2]][count1])
            count2+=1
        PokeDex.append(Pokemon(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9]))
        count1+=1
        printPoke(PokeDex[count1-1])

if __name__ == '__main__':
    makeDex()