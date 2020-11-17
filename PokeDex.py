import pandas as pd

class Pokemon:
  def __init__(self,name,type,move1,move2,move3,move4,move1type,move2type,move3type,move4type,hp_stat,atk_stat,def_stat,spec_stat,speed_stat):
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
    self.hp_stat = hp_stat
    self.atk_stat = atk_stat
    self.def_stat = def_stat
    self.spec_stat = spec_stat
    self.speed_stat = speed_stat


class Move:
    def __init__(self, name, name_caps, move_type, category, power, accuracy, max_pp):
        self.name = name
        self.name_caps = name_caps
        self.move_type = move_type
        self.category = category
        self.power = power
        self.accuracy = accuracy
        self.max_pp = max_pp


def printPoke(Pokemon):
    print(Pokemon.name + " ~ " + Pokemon.type + "\n")
    print(Pokemon.move1 + " ~ " + Pokemon.move1type + "\n")
    print(Pokemon.move2 + " ~ " + Pokemon.move2type + "\n")
    print(Pokemon.move3 + " ~ " + Pokemon.move3type + "\n")
    print(Pokemon.move4 + " ~ " + Pokemon.move4type + "\n")
    print("Stats= HP:" + str(Pokemon.hp_stat)
          + " ATK:" + str(Pokemon.atk_stat)
          + " DEF:" + str(Pokemon.def_stat)
          + " SPE:" + str(Pokemon.spec_stat)
          + " SPD:" + str(Pokemon.speed_stat) + "\n\n")


def print_move(move):
    print("Move: " + move.name + "\n" + "Type: " + move.move_type + " (" + move.category + ")")
    print("Power: [" + str(move.power) + "] " + "Accuracy: [" + str(move.accuracy) + "] " + "PP: [" + str(move.max_pp) + "/" + str(move.max_pp) + "]")


def makeDex():
    PokeDex = []
    df = pd.read_csv('PokeDex.csv')
    index = "Name", "Types", "Move1", "Move2", "Move3", "Move4", "Move1Type", "Move2Type", "Move3Type", "Move4Type", "HP", "Attack", "Defense", "Special", "Speed"
    # print(df)
    # print(df[index[0]][0])
    count1=0
    while (count1 < 150):
        count2 = 0
        data = []
        while (count2 < 15):
            data.append(df[index[count2]][count1])
            count2+=1
        PokeDex.append(Pokemon(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14]))
        count1+=1
        # printPoke(PokeDex[count1-1])
    return PokeDex  # return value for testing purposes, sorry Brandon. -Ian


def make_moves():
    """ Ian's move loading function """
    moves = []
    df = pd.read_csv('Movelist.csv')
    index = "Name", "NameCaps", "Type", "Category", "Power", "Accuracy", "MaxPP"
    i = 0
    # print((df[index[0]]))
    while i < len(df[index[0]]):
        j = 0
        move_data = []
        while j < len(index):
            move_data.append(df[index[j]][i])
            j += 1
        moves.append(Move(move_data[0], move_data[1], move_data[2], move_data[3], move_data[4], move_data[5], move_data[6]))
        i += 1
    return moves


if __name__ == '__main__':
    makeDex()