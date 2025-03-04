\ Model sudoku
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: indicateurs[1,1,9] = 1
 R1: indicateurs[2,1,8] = 1
 R2: indicateurs[5,1,3] = 1
 R3: indicateurs[9,1,6] = 1
 R4: indicateurs[2,2,5] = 1
 R5: indicateurs[3,2,7] = 1
 R6: indicateurs[4,2,6] = 1
 R7: indicateurs[4,3,8] = 1
 R8: indicateurs[8,3,4] = 1
 R9: indicateurs[9,3,5] = 1
 R10: indicateurs[1,4,3] = 1
 R11: indicateurs[2,4,7] = 1
 R12: indicateurs[3,4,8] = 1
 R13: indicateurs[5,4,6] = 1
 R14: indicateurs[8,4,1] = 1
 R15: indicateurs[1,5,4] = 1
 R16: indicateurs[3,5,9] = 1
 R17: indicateurs[5,5,8] = 1
 R18: indicateurs[6,5,5] = 1
 R19: indicateurs[9,5,7] = 1
 R20: indicateurs[3,6,1] = 1
 R21: indicateurs[5,6,7] = 1
 R22: indicateurs[6,6,4] = 1
 R23: indicateurs[7,6,8] = 1
 R24: indicateurs[8,6,2] = 1
 R25: indicateurs[9,6,9] = 1
 R26: indicateurs[2,7,3] = 1
 R27: indicateurs[3,7,2] = 1
 R28: indicateurs[4,7,4] = 1
 R29: indicateurs[8,7,5] = 1
 R30: indicateurs[9,7,1] = 1
 R31: indicateurs[1,8,7] = 1
 R32: indicateurs[2,8,4] = 1
 R33: indicateurs[5,8,1] = 1
 R34: indicateurs[2,9,9] = 1
 R35: indicateurs[5,9,5] = 1
 R36: indicateurs[6,9,3] = 1
 R37: indicateurs[9,9,2] = 1
 R38: indicateurs[1,1,1] + indicateurs[1,2,1] + indicateurs[1,3,1]
   + indicateurs[1,4,1] + indicateurs[1,5,1] + indicateurs[1,6,1]
   + indicateurs[1,7,1] + indicateurs[1,8,1] + indicateurs[1,9,1] = 1
 R39: indicateurs[1,1,2] + indicateurs[1,2,2] + indicateurs[1,3,2]
   + indicateurs[1,4,2] + indicateurs[1,5,2] + indicateurs[1,6,2]
   + indicateurs[1,7,2] + indicateurs[1,8,2] + indicateurs[1,9,2] = 1
 R40: indicateurs[1,1,3] + indicateurs[1,2,3] + indicateurs[1,3,3]
   + indicateurs[1,4,3] + indicateurs[1,5,3] + indicateurs[1,6,3]
   + indicateurs[1,7,3] + indicateurs[1,8,3] + indicateurs[1,9,3] = 1
 R41: indicateurs[1,1,4] + indicateurs[1,2,4] + indicateurs[1,3,4]
   + indicateurs[1,4,4] + indicateurs[1,5,4] + indicateurs[1,6,4]
   + indicateurs[1,7,4] + indicateurs[1,8,4] + indicateurs[1,9,4] = 1
 R42: indicateurs[1,1,5] + indicateurs[1,2,5] + indicateurs[1,3,5]
   + indicateurs[1,4,5] + indicateurs[1,5,5] + indicateurs[1,6,5]
   + indicateurs[1,7,5] + indicateurs[1,8,5] + indicateurs[1,9,5] = 1
 R43: indicateurs[1,1,6] + indicateurs[1,2,6] + indicateurs[1,3,6]
   + indicateurs[1,4,6] + indicateurs[1,5,6] + indicateurs[1,6,6]
   + indicateurs[1,7,6] + indicateurs[1,8,6] + indicateurs[1,9,6] = 1
 R44: indicateurs[1,1,7] + indicateurs[1,2,7] + indicateurs[1,3,7]
   + indicateurs[1,4,7] + indicateurs[1,5,7] + indicateurs[1,6,7]
   + indicateurs[1,7,7] + indicateurs[1,8,7] + indicateurs[1,9,7] = 1
 R45: indicateurs[1,1,8] + indicateurs[1,2,8] + indicateurs[1,3,8]
   + indicateurs[1,4,8] + indicateurs[1,5,8] + indicateurs[1,6,8]
   + indicateurs[1,7,8] + indicateurs[1,8,8] + indicateurs[1,9,8] = 1
 R46: indicateurs[1,1,9] + indicateurs[1,2,9] + indicateurs[1,3,9]
   + indicateurs[1,4,9] + indicateurs[1,5,9] + indicateurs[1,6,9]
   + indicateurs[1,7,9] + indicateurs[1,8,9] + indicateurs[1,9,9] = 1
 R47: indicateurs[2,1,1] + indicateurs[2,2,1] + indicateurs[2,3,1]
   + indicateurs[2,4,1] + indicateurs[2,5,1] + indicateurs[2,6,1]
   + indicateurs[2,7,1] + indicateurs[2,8,1] + indicateurs[2,9,1] = 1
 R48: indicateurs[2,1,2] + indicateurs[2,2,2] + indicateurs[2,3,2]
   + indicateurs[2,4,2] + indicateurs[2,5,2] + indicateurs[2,6,2]
   + indicateurs[2,7,2] + indicateurs[2,8,2] + indicateurs[2,9,2] = 1
 R49: indicateurs[2,1,3] + indicateurs[2,2,3] + indicateurs[2,3,3]
   + indicateurs[2,4,3] + indicateurs[2,5,3] + indicateurs[2,6,3]
   + indicateurs[2,7,3] + indicateurs[2,8,3] + indicateurs[2,9,3] = 1
 R50: indicateurs[2,1,4] + indicateurs[2,2,4] + indicateurs[2,3,4]
   + indicateurs[2,4,4] + indicateurs[2,5,4] + indicateurs[2,6,4]
   + indicateurs[2,7,4] + indicateurs[2,8,4] + indicateurs[2,9,4] = 1
 R51: indicateurs[2,1,5] + indicateurs[2,2,5] + indicateurs[2,3,5]
   + indicateurs[2,4,5] + indicateurs[2,5,5] + indicateurs[2,6,5]
   + indicateurs[2,7,5] + indicateurs[2,8,5] + indicateurs[2,9,5] = 1
 R52: indicateurs[2,1,6] + indicateurs[2,2,6] + indicateurs[2,3,6]
   + indicateurs[2,4,6] + indicateurs[2,5,6] + indicateurs[2,6,6]
   + indicateurs[2,7,6] + indicateurs[2,8,6] + indicateurs[2,9,6] = 1
 R53: indicateurs[2,1,7] + indicateurs[2,2,7] + indicateurs[2,3,7]
   + indicateurs[2,4,7] + indicateurs[2,5,7] + indicateurs[2,6,7]
   + indicateurs[2,7,7] + indicateurs[2,8,7] + indicateurs[2,9,7] = 1
 R54: indicateurs[2,1,8] + indicateurs[2,2,8] + indicateurs[2,3,8]
   + indicateurs[2,4,8] + indicateurs[2,5,8] + indicateurs[2,6,8]
   + indicateurs[2,7,8] + indicateurs[2,8,8] + indicateurs[2,9,8] = 1
 R55: indicateurs[2,1,9] + indicateurs[2,2,9] + indicateurs[2,3,9]
   + indicateurs[2,4,9] + indicateurs[2,5,9] + indicateurs[2,6,9]
   + indicateurs[2,7,9] + indicateurs[2,8,9] + indicateurs[2,9,9] = 1
 R56: indicateurs[3,1,1] + indicateurs[3,2,1] + indicateurs[3,3,1]
   + indicateurs[3,4,1] + indicateurs[3,5,1] + indicateurs[3,6,1]
   + indicateurs[3,7,1] + indicateurs[3,8,1] + indicateurs[3,9,1] = 1
 R57: indicateurs[3,1,2] + indicateurs[3,2,2] + indicateurs[3,3,2]
   + indicateurs[3,4,2] + indicateurs[3,5,2] + indicateurs[3,6,2]
   + indicateurs[3,7,2] + indicateurs[3,8,2] + indicateurs[3,9,2] = 1
 R58: indicateurs[3,1,3] + indicateurs[3,2,3] + indicateurs[3,3,3]
   + indicateurs[3,4,3] + indicateurs[3,5,3] + indicateurs[3,6,3]
   + indicateurs[3,7,3] + indicateurs[3,8,3] + indicateurs[3,9,3] = 1
 R59: indicateurs[3,1,4] + indicateurs[3,2,4] + indicateurs[3,3,4]
   + indicateurs[3,4,4] + indicateurs[3,5,4] + indicateurs[3,6,4]
   + indicateurs[3,7,4] + indicateurs[3,8,4] + indicateurs[3,9,4] = 1
 R60: indicateurs[3,1,5] + indicateurs[3,2,5] + indicateurs[3,3,5]
   + indicateurs[3,4,5] + indicateurs[3,5,5] + indicateurs[3,6,5]
   + indicateurs[3,7,5] + indicateurs[3,8,5] + indicateurs[3,9,5] = 1
 R61: indicateurs[3,1,6] + indicateurs[3,2,6] + indicateurs[3,3,6]
   + indicateurs[3,4,6] + indicateurs[3,5,6] + indicateurs[3,6,6]
   + indicateurs[3,7,6] + indicateurs[3,8,6] + indicateurs[3,9,6] = 1
 R62: indicateurs[3,1,7] + indicateurs[3,2,7] + indicateurs[3,3,7]
   + indicateurs[3,4,7] + indicateurs[3,5,7] + indicateurs[3,6,7]
   + indicateurs[3,7,7] + indicateurs[3,8,7] + indicateurs[3,9,7] = 1
 R63: indicateurs[3,1,8] + indicateurs[3,2,8] + indicateurs[3,3,8]
   + indicateurs[3,4,8] + indicateurs[3,5,8] + indicateurs[3,6,8]
   + indicateurs[3,7,8] + indicateurs[3,8,8] + indicateurs[3,9,8] = 1
 R64: indicateurs[3,1,9] + indicateurs[3,2,9] + indicateurs[3,3,9]
   + indicateurs[3,4,9] + indicateurs[3,5,9] + indicateurs[3,6,9]
   + indicateurs[3,7,9] + indicateurs[3,8,9] + indicateurs[3,9,9] = 1
 R65: indicateurs[4,1,1] + indicateurs[4,2,1] + indicateurs[4,3,1]
   + indicateurs[4,4,1] + indicateurs[4,5,1] + indicateurs[4,6,1]
   + indicateurs[4,7,1] + indicateurs[4,8,1] + indicateurs[4,9,1] = 1
 R66: indicateurs[4,1,2] + indicateurs[4,2,2] + indicateurs[4,3,2]
   + indicateurs[4,4,2] + indicateurs[4,5,2] + indicateurs[4,6,2]
   + indicateurs[4,7,2] + indicateurs[4,8,2] + indicateurs[4,9,2] = 1
 R67: indicateurs[4,1,3] + indicateurs[4,2,3] + indicateurs[4,3,3]
   + indicateurs[4,4,3] + indicateurs[4,5,3] + indicateurs[4,6,3]
   + indicateurs[4,7,3] + indicateurs[4,8,3] + indicateurs[4,9,3] = 1
 R68: indicateurs[4,1,4] + indicateurs[4,2,4] + indicateurs[4,3,4]
   + indicateurs[4,4,4] + indicateurs[4,5,4] + indicateurs[4,6,4]
   + indicateurs[4,7,4] + indicateurs[4,8,4] + indicateurs[4,9,4] = 1
 R69: indicateurs[4,1,5] + indicateurs[4,2,5] + indicateurs[4,3,5]
   + indicateurs[4,4,5] + indicateurs[4,5,5] + indicateurs[4,6,5]
   + indicateurs[4,7,5] + indicateurs[4,8,5] + indicateurs[4,9,5] = 1
 R70: indicateurs[4,1,6] + indicateurs[4,2,6] + indicateurs[4,3,6]
   + indicateurs[4,4,6] + indicateurs[4,5,6] + indicateurs[4,6,6]
   + indicateurs[4,7,6] + indicateurs[4,8,6] + indicateurs[4,9,6] = 1
 R71: indicateurs[4,1,7] + indicateurs[4,2,7] + indicateurs[4,3,7]
   + indicateurs[4,4,7] + indicateurs[4,5,7] + indicateurs[4,6,7]
   + indicateurs[4,7,7] + indicateurs[4,8,7] + indicateurs[4,9,7] = 1
 R72: indicateurs[4,1,8] + indicateurs[4,2,8] + indicateurs[4,3,8]
   + indicateurs[4,4,8] + indicateurs[4,5,8] + indicateurs[4,6,8]
   + indicateurs[4,7,8] + indicateurs[4,8,8] + indicateurs[4,9,8] = 1
 R73: indicateurs[4,1,9] + indicateurs[4,2,9] + indicateurs[4,3,9]
   + indicateurs[4,4,9] + indicateurs[4,5,9] + indicateurs[4,6,9]
   + indicateurs[4,7,9] + indicateurs[4,8,9] + indicateurs[4,9,9] = 1
 R74: indicateurs[5,1,1] + indicateurs[5,2,1] + indicateurs[5,3,1]
   + indicateurs[5,4,1] + indicateurs[5,5,1] + indicateurs[5,6,1]
   + indicateurs[5,7,1] + indicateurs[5,8,1] + indicateurs[5,9,1] = 1
 R75: indicateurs[5,1,2] + indicateurs[5,2,2] + indicateurs[5,3,2]
   + indicateurs[5,4,2] + indicateurs[5,5,2] + indicateurs[5,6,2]
   + indicateurs[5,7,2] + indicateurs[5,8,2] + indicateurs[5,9,2] = 1
 R76: indicateurs[5,1,3] + indicateurs[5,2,3] + indicateurs[5,3,3]
   + indicateurs[5,4,3] + indicateurs[5,5,3] + indicateurs[5,6,3]
   + indicateurs[5,7,3] + indicateurs[5,8,3] + indicateurs[5,9,3] = 1
 R77: indicateurs[5,1,4] + indicateurs[5,2,4] + indicateurs[5,3,4]
   + indicateurs[5,4,4] + indicateurs[5,5,4] + indicateurs[5,6,4]
   + indicateurs[5,7,4] + indicateurs[5,8,4] + indicateurs[5,9,4] = 1
 R78: indicateurs[5,1,5] + indicateurs[5,2,5] + indicateurs[5,3,5]
   + indicateurs[5,4,5] + indicateurs[5,5,5] + indicateurs[5,6,5]
   + indicateurs[5,7,5] + indicateurs[5,8,5] + indicateurs[5,9,5] = 1
 R79: indicateurs[5,1,6] + indicateurs[5,2,6] + indicateurs[5,3,6]
   + indicateurs[5,4,6] + indicateurs[5,5,6] + indicateurs[5,6,6]
   + indicateurs[5,7,6] + indicateurs[5,8,6] + indicateurs[5,9,6] = 1
 R80: indicateurs[5,1,7] + indicateurs[5,2,7] + indicateurs[5,3,7]
   + indicateurs[5,4,7] + indicateurs[5,5,7] + indicateurs[5,6,7]
   + indicateurs[5,7,7] + indicateurs[5,8,7] + indicateurs[5,9,7] = 1
 R81: indicateurs[5,1,8] + indicateurs[5,2,8] + indicateurs[5,3,8]
   + indicateurs[5,4,8] + indicateurs[5,5,8] + indicateurs[5,6,8]
   + indicateurs[5,7,8] + indicateurs[5,8,8] + indicateurs[5,9,8] = 1
 R82: indicateurs[5,1,9] + indicateurs[5,2,9] + indicateurs[5,3,9]
   + indicateurs[5,4,9] + indicateurs[5,5,9] + indicateurs[5,6,9]
   + indicateurs[5,7,9] + indicateurs[5,8,9] + indicateurs[5,9,9] = 1
 R83: indicateurs[6,1,1] + indicateurs[6,2,1] + indicateurs[6,3,1]
   + indicateurs[6,4,1] + indicateurs[6,5,1] + indicateurs[6,6,1]
   + indicateurs[6,7,1] + indicateurs[6,8,1] + indicateurs[6,9,1] = 1
 R84: indicateurs[6,1,2] + indicateurs[6,2,2] + indicateurs[6,3,2]
   + indicateurs[6,4,2] + indicateurs[6,5,2] + indicateurs[6,6,2]
   + indicateurs[6,7,2] + indicateurs[6,8,2] + indicateurs[6,9,2] = 1
 R85: indicateurs[6,1,3] + indicateurs[6,2,3] + indicateurs[6,3,3]
   + indicateurs[6,4,3] + indicateurs[6,5,3] + indicateurs[6,6,3]
   + indicateurs[6,7,3] + indicateurs[6,8,3] + indicateurs[6,9,3] = 1
 R86: indicateurs[6,1,4] + indicateurs[6,2,4] + indicateurs[6,3,4]
   + indicateurs[6,4,4] + indicateurs[6,5,4] + indicateurs[6,6,4]
   + indicateurs[6,7,4] + indicateurs[6,8,4] + indicateurs[6,9,4] = 1
 R87: indicateurs[6,1,5] + indicateurs[6,2,5] + indicateurs[6,3,5]
   + indicateurs[6,4,5] + indicateurs[6,5,5] + indicateurs[6,6,5]
   + indicateurs[6,7,5] + indicateurs[6,8,5] + indicateurs[6,9,5] = 1
 R88: indicateurs[6,1,6] + indicateurs[6,2,6] + indicateurs[6,3,6]
   + indicateurs[6,4,6] + indicateurs[6,5,6] + indicateurs[6,6,6]
   + indicateurs[6,7,6] + indicateurs[6,8,6] + indicateurs[6,9,6] = 1
 R89: indicateurs[6,1,7] + indicateurs[6,2,7] + indicateurs[6,3,7]
   + indicateurs[6,4,7] + indicateurs[6,5,7] + indicateurs[6,6,7]
   + indicateurs[6,7,7] + indicateurs[6,8,7] + indicateurs[6,9,7] = 1
 R90: indicateurs[6,1,8] + indicateurs[6,2,8] + indicateurs[6,3,8]
   + indicateurs[6,4,8] + indicateurs[6,5,8] + indicateurs[6,6,8]
   + indicateurs[6,7,8] + indicateurs[6,8,8] + indicateurs[6,9,8] = 1
 R91: indicateurs[6,1,9] + indicateurs[6,2,9] + indicateurs[6,3,9]
   + indicateurs[6,4,9] + indicateurs[6,5,9] + indicateurs[6,6,9]
   + indicateurs[6,7,9] + indicateurs[6,8,9] + indicateurs[6,9,9] = 1
 R92: indicateurs[7,1,1] + indicateurs[7,2,1] + indicateurs[7,3,1]
   + indicateurs[7,4,1] + indicateurs[7,5,1] + indicateurs[7,6,1]
   + indicateurs[7,7,1] + indicateurs[7,8,1] + indicateurs[7,9,1] = 1
 R93: indicateurs[7,1,2] + indicateurs[7,2,2] + indicateurs[7,3,2]
   + indicateurs[7,4,2] + indicateurs[7,5,2] + indicateurs[7,6,2]
   + indicateurs[7,7,2] + indicateurs[7,8,2] + indicateurs[7,9,2] = 1
 R94: indicateurs[7,1,3] + indicateurs[7,2,3] + indicateurs[7,3,3]
   + indicateurs[7,4,3] + indicateurs[7,5,3] + indicateurs[7,6,3]
   + indicateurs[7,7,3] + indicateurs[7,8,3] + indicateurs[7,9,3] = 1
 R95: indicateurs[7,1,4] + indicateurs[7,2,4] + indicateurs[7,3,4]
   + indicateurs[7,4,4] + indicateurs[7,5,4] + indicateurs[7,6,4]
   + indicateurs[7,7,4] + indicateurs[7,8,4] + indicateurs[7,9,4] = 1
 R96: indicateurs[7,1,5] + indicateurs[7,2,5] + indicateurs[7,3,5]
   + indicateurs[7,4,5] + indicateurs[7,5,5] + indicateurs[7,6,5]
   + indicateurs[7,7,5] + indicateurs[7,8,5] + indicateurs[7,9,5] = 1
 R97: indicateurs[7,1,6] + indicateurs[7,2,6] + indicateurs[7,3,6]
   + indicateurs[7,4,6] + indicateurs[7,5,6] + indicateurs[7,6,6]
   + indicateurs[7,7,6] + indicateurs[7,8,6] + indicateurs[7,9,6] = 1
 R98: indicateurs[7,1,7] + indicateurs[7,2,7] + indicateurs[7,3,7]
   + indicateurs[7,4,7] + indicateurs[7,5,7] + indicateurs[7,6,7]
   + indicateurs[7,7,7] + indicateurs[7,8,7] + indicateurs[7,9,7] = 1
 R99: indicateurs[7,1,8] + indicateurs[7,2,8] + indicateurs[7,3,8]
   + indicateurs[7,4,8] + indicateurs[7,5,8] + indicateurs[7,6,8]
   + indicateurs[7,7,8] + indicateurs[7,8,8] + indicateurs[7,9,8] = 1
 R100: indicateurs[7,1,9] + indicateurs[7,2,9] + indicateurs[7,3,9]
   + indicateurs[7,4,9] + indicateurs[7,5,9] + indicateurs[7,6,9]
   + indicateurs[7,7,9] + indicateurs[7,8,9] + indicateurs[7,9,9] = 1
 R101: indicateurs[8,1,1] + indicateurs[8,2,1] + indicateurs[8,3,1]
   + indicateurs[8,4,1] + indicateurs[8,5,1] + indicateurs[8,6,1]
   + indicateurs[8,7,1] + indicateurs[8,8,1] + indicateurs[8,9,1] = 1
 R102: indicateurs[8,1,2] + indicateurs[8,2,2] + indicateurs[8,3,2]
   + indicateurs[8,4,2] + indicateurs[8,5,2] + indicateurs[8,6,2]
   + indicateurs[8,7,2] + indicateurs[8,8,2] + indicateurs[8,9,2] = 1
 R103: indicateurs[8,1,3] + indicateurs[8,2,3] + indicateurs[8,3,3]
   + indicateurs[8,4,3] + indicateurs[8,5,3] + indicateurs[8,6,3]
   + indicateurs[8,7,3] + indicateurs[8,8,3] + indicateurs[8,9,3] = 1
 R104: indicateurs[8,1,4] + indicateurs[8,2,4] + indicateurs[8,3,4]
   + indicateurs[8,4,4] + indicateurs[8,5,4] + indicateurs[8,6,4]
   + indicateurs[8,7,4] + indicateurs[8,8,4] + indicateurs[8,9,4] = 1
 R105: indicateurs[8,1,5] + indicateurs[8,2,5] + indicateurs[8,3,5]
   + indicateurs[8,4,5] + indicateurs[8,5,5] + indicateurs[8,6,5]
   + indicateurs[8,7,5] + indicateurs[8,8,5] + indicateurs[8,9,5] = 1
 R106: indicateurs[8,1,6] + indicateurs[8,2,6] + indicateurs[8,3,6]
   + indicateurs[8,4,6] + indicateurs[8,5,6] + indicateurs[8,6,6]
   + indicateurs[8,7,6] + indicateurs[8,8,6] + indicateurs[8,9,6] = 1
 R107: indicateurs[8,1,7] + indicateurs[8,2,7] + indicateurs[8,3,7]
   + indicateurs[8,4,7] + indicateurs[8,5,7] + indicateurs[8,6,7]
   + indicateurs[8,7,7] + indicateurs[8,8,7] + indicateurs[8,9,7] = 1
 R108: indicateurs[8,1,8] + indicateurs[8,2,8] + indicateurs[8,3,8]
   + indicateurs[8,4,8] + indicateurs[8,5,8] + indicateurs[8,6,8]
   + indicateurs[8,7,8] + indicateurs[8,8,8] + indicateurs[8,9,8] = 1
 R109: indicateurs[8,1,9] + indicateurs[8,2,9] + indicateurs[8,3,9]
   + indicateurs[8,4,9] + indicateurs[8,5,9] + indicateurs[8,6,9]
   + indicateurs[8,7,9] + indicateurs[8,8,9] + indicateurs[8,9,9] = 1
 R110: indicateurs[9,1,1] + indicateurs[9,2,1] + indicateurs[9,3,1]
   + indicateurs[9,4,1] + indicateurs[9,5,1] + indicateurs[9,6,1]
   + indicateurs[9,7,1] + indicateurs[9,8,1] + indicateurs[9,9,1] = 1
 R111: indicateurs[9,1,2] + indicateurs[9,2,2] + indicateurs[9,3,2]
   + indicateurs[9,4,2] + indicateurs[9,5,2] + indicateurs[9,6,2]
   + indicateurs[9,7,2] + indicateurs[9,8,2] + indicateurs[9,9,2] = 1
 R112: indicateurs[9,1,3] + indicateurs[9,2,3] + indicateurs[9,3,3]
   + indicateurs[9,4,3] + indicateurs[9,5,3] + indicateurs[9,6,3]
   + indicateurs[9,7,3] + indicateurs[9,8,3] + indicateurs[9,9,3] = 1
 R113: indicateurs[9,1,4] + indicateurs[9,2,4] + indicateurs[9,3,4]
   + indicateurs[9,4,4] + indicateurs[9,5,4] + indicateurs[9,6,4]
   + indicateurs[9,7,4] + indicateurs[9,8,4] + indicateurs[9,9,4] = 1
 R114: indicateurs[9,1,5] + indicateurs[9,2,5] + indicateurs[9,3,5]
   + indicateurs[9,4,5] + indicateurs[9,5,5] + indicateurs[9,6,5]
   + indicateurs[9,7,5] + indicateurs[9,8,5] + indicateurs[9,9,5] = 1
 R115: indicateurs[9,1,6] + indicateurs[9,2,6] + indicateurs[9,3,6]
   + indicateurs[9,4,6] + indicateurs[9,5,6] + indicateurs[9,6,6]
   + indicateurs[9,7,6] + indicateurs[9,8,6] + indicateurs[9,9,6] = 1
 R116: indicateurs[9,1,7] + indicateurs[9,2,7] + indicateurs[9,3,7]
   + indicateurs[9,4,7] + indicateurs[9,5,7] + indicateurs[9,6,7]
   + indicateurs[9,7,7] + indicateurs[9,8,7] + indicateurs[9,9,7] = 1
 R117: indicateurs[9,1,8] + indicateurs[9,2,8] + indicateurs[9,3,8]
   + indicateurs[9,4,8] + indicateurs[9,5,8] + indicateurs[9,6,8]
   + indicateurs[9,7,8] + indicateurs[9,8,8] + indicateurs[9,9,8] = 1
 R118: indicateurs[9,1,9] + indicateurs[9,2,9] + indicateurs[9,3,9]
   + indicateurs[9,4,9] + indicateurs[9,5,9] + indicateurs[9,6,9]
   + indicateurs[9,7,9] + indicateurs[9,8,9] + indicateurs[9,9,9] = 1
 R119: indicateurs[1,1,1] + indicateurs[2,1,1] + indicateurs[3,1,1]
   + indicateurs[4,1,1] + indicateurs[5,1,1] + indicateurs[6,1,1]
   + indicateurs[7,1,1] + indicateurs[8,1,1] + indicateurs[9,1,1] = 1
 R120: indicateurs[1,1,2] + indicateurs[2,1,2] + indicateurs[3,1,2]
   + indicateurs[4,1,2] + indicateurs[5,1,2] + indicateurs[6,1,2]
   + indicateurs[7,1,2] + indicateurs[8,1,2] + indicateurs[9,1,2] = 1
 R121: indicateurs[1,1,3] + indicateurs[2,1,3] + indicateurs[3,1,3]
   + indicateurs[4,1,3] + indicateurs[5,1,3] + indicateurs[6,1,3]
   + indicateurs[7,1,3] + indicateurs[8,1,3] + indicateurs[9,1,3] = 1
 R122: indicateurs[1,1,4] + indicateurs[2,1,4] + indicateurs[3,1,4]
   + indicateurs[4,1,4] + indicateurs[5,1,4] + indicateurs[6,1,4]
   + indicateurs[7,1,4] + indicateurs[8,1,4] + indicateurs[9,1,4] = 1
 R123: indicateurs[1,1,5] + indicateurs[2,1,5] + indicateurs[3,1,5]
   + indicateurs[4,1,5] + indicateurs[5,1,5] + indicateurs[6,1,5]
   + indicateurs[7,1,5] + indicateurs[8,1,5] + indicateurs[9,1,5] = 1
 R124: indicateurs[1,1,6] + indicateurs[2,1,6] + indicateurs[3,1,6]
   + indicateurs[4,1,6] + indicateurs[5,1,6] + indicateurs[6,1,6]
   + indicateurs[7,1,6] + indicateurs[8,1,6] + indicateurs[9,1,6] = 1
 R125: indicateurs[1,1,7] + indicateurs[2,1,7] + indicateurs[3,1,7]
   + indicateurs[4,1,7] + indicateurs[5,1,7] + indicateurs[6,1,7]
   + indicateurs[7,1,7] + indicateurs[8,1,7] + indicateurs[9,1,7] = 1
 R126: indicateurs[1,1,8] + indicateurs[2,1,8] + indicateurs[3,1,8]
   + indicateurs[4,1,8] + indicateurs[5,1,8] + indicateurs[6,1,8]
   + indicateurs[7,1,8] + indicateurs[8,1,8] + indicateurs[9,1,8] = 1
 R127: indicateurs[1,1,9] + indicateurs[2,1,9] + indicateurs[3,1,9]
   + indicateurs[4,1,9] + indicateurs[5,1,9] + indicateurs[6,1,9]
   + indicateurs[7,1,9] + indicateurs[8,1,9] + indicateurs[9,1,9] = 1
 R128: indicateurs[1,2,1] + indicateurs[2,2,1] + indicateurs[3,2,1]
   + indicateurs[4,2,1] + indicateurs[5,2,1] + indicateurs[6,2,1]
   + indicateurs[7,2,1] + indicateurs[8,2,1] + indicateurs[9,2,1] = 1
 R129: indicateurs[1,2,2] + indicateurs[2,2,2] + indicateurs[3,2,2]
   + indicateurs[4,2,2] + indicateurs[5,2,2] + indicateurs[6,2,2]
   + indicateurs[7,2,2] + indicateurs[8,2,2] + indicateurs[9,2,2] = 1
 R130: indicateurs[1,2,3] + indicateurs[2,2,3] + indicateurs[3,2,3]
   + indicateurs[4,2,3] + indicateurs[5,2,3] + indicateurs[6,2,3]
   + indicateurs[7,2,3] + indicateurs[8,2,3] + indicateurs[9,2,3] = 1
 R131: indicateurs[1,2,4] + indicateurs[2,2,4] + indicateurs[3,2,4]
   + indicateurs[4,2,4] + indicateurs[5,2,4] + indicateurs[6,2,4]
   + indicateurs[7,2,4] + indicateurs[8,2,4] + indicateurs[9,2,4] = 1
 R132: indicateurs[1,2,5] + indicateurs[2,2,5] + indicateurs[3,2,5]
   + indicateurs[4,2,5] + indicateurs[5,2,5] + indicateurs[6,2,5]
   + indicateurs[7,2,5] + indicateurs[8,2,5] + indicateurs[9,2,5] = 1
 R133: indicateurs[1,2,6] + indicateurs[2,2,6] + indicateurs[3,2,6]
   + indicateurs[4,2,6] + indicateurs[5,2,6] + indicateurs[6,2,6]
   + indicateurs[7,2,6] + indicateurs[8,2,6] + indicateurs[9,2,6] = 1
 R134: indicateurs[1,2,7] + indicateurs[2,2,7] + indicateurs[3,2,7]
   + indicateurs[4,2,7] + indicateurs[5,2,7] + indicateurs[6,2,7]
   + indicateurs[7,2,7] + indicateurs[8,2,7] + indicateurs[9,2,7] = 1
 R135: indicateurs[1,2,8] + indicateurs[2,2,8] + indicateurs[3,2,8]
   + indicateurs[4,2,8] + indicateurs[5,2,8] + indicateurs[6,2,8]
   + indicateurs[7,2,8] + indicateurs[8,2,8] + indicateurs[9,2,8] = 1
 R136: indicateurs[1,2,9] + indicateurs[2,2,9] + indicateurs[3,2,9]
   + indicateurs[4,2,9] + indicateurs[5,2,9] + indicateurs[6,2,9]
   + indicateurs[7,2,9] + indicateurs[8,2,9] + indicateurs[9,2,9] = 1
 R137: indicateurs[1,3,1] + indicateurs[2,3,1] + indicateurs[3,3,1]
   + indicateurs[4,3,1] + indicateurs[5,3,1] + indicateurs[6,3,1]
   + indicateurs[7,3,1] + indicateurs[8,3,1] + indicateurs[9,3,1] = 1
 R138: indicateurs[1,3,2] + indicateurs[2,3,2] + indicateurs[3,3,2]
   + indicateurs[4,3,2] + indicateurs[5,3,2] + indicateurs[6,3,2]
   + indicateurs[7,3,2] + indicateurs[8,3,2] + indicateurs[9,3,2] = 1
 R139: indicateurs[1,3,3] + indicateurs[2,3,3] + indicateurs[3,3,3]
   + indicateurs[4,3,3] + indicateurs[5,3,3] + indicateurs[6,3,3]
   + indicateurs[7,3,3] + indicateurs[8,3,3] + indicateurs[9,3,3] = 1
 R140: indicateurs[1,3,4] + indicateurs[2,3,4] + indicateurs[3,3,4]
   + indicateurs[4,3,4] + indicateurs[5,3,4] + indicateurs[6,3,4]
   + indicateurs[7,3,4] + indicateurs[8,3,4] + indicateurs[9,3,4] = 1
 R141: indicateurs[1,3,5] + indicateurs[2,3,5] + indicateurs[3,3,5]
   + indicateurs[4,3,5] + indicateurs[5,3,5] + indicateurs[6,3,5]
   + indicateurs[7,3,5] + indicateurs[8,3,5] + indicateurs[9,3,5] = 1
 R142: indicateurs[1,3,6] + indicateurs[2,3,6] + indicateurs[3,3,6]
   + indicateurs[4,3,6] + indicateurs[5,3,6] + indicateurs[6,3,6]
   + indicateurs[7,3,6] + indicateurs[8,3,6] + indicateurs[9,3,6] = 1
 R143: indicateurs[1,3,7] + indicateurs[2,3,7] + indicateurs[3,3,7]
   + indicateurs[4,3,7] + indicateurs[5,3,7] + indicateurs[6,3,7]
   + indicateurs[7,3,7] + indicateurs[8,3,7] + indicateurs[9,3,7] = 1
 R144: indicateurs[1,3,8] + indicateurs[2,3,8] + indicateurs[3,3,8]
   + indicateurs[4,3,8] + indicateurs[5,3,8] + indicateurs[6,3,8]
   + indicateurs[7,3,8] + indicateurs[8,3,8] + indicateurs[9,3,8] = 1
 R145: indicateurs[1,3,9] + indicateurs[2,3,9] + indicateurs[3,3,9]
   + indicateurs[4,3,9] + indicateurs[5,3,9] + indicateurs[6,3,9]
   + indicateurs[7,3,9] + indicateurs[8,3,9] + indicateurs[9,3,9] = 1
 R146: indicateurs[1,4,1] + indicateurs[2,4,1] + indicateurs[3,4,1]
   + indicateurs[4,4,1] + indicateurs[5,4,1] + indicateurs[6,4,1]
   + indicateurs[7,4,1] + indicateurs[8,4,1] + indicateurs[9,4,1] = 1
 R147: indicateurs[1,4,2] + indicateurs[2,4,2] + indicateurs[3,4,2]
   + indicateurs[4,4,2] + indicateurs[5,4,2] + indicateurs[6,4,2]
   + indicateurs[7,4,2] + indicateurs[8,4,2] + indicateurs[9,4,2] = 1
 R148: indicateurs[1,4,3] + indicateurs[2,4,3] + indicateurs[3,4,3]
   + indicateurs[4,4,3] + indicateurs[5,4,3] + indicateurs[6,4,3]
   + indicateurs[7,4,3] + indicateurs[8,4,3] + indicateurs[9,4,3] = 1
 R149: indicateurs[1,4,4] + indicateurs[2,4,4] + indicateurs[3,4,4]
   + indicateurs[4,4,4] + indicateurs[5,4,4] + indicateurs[6,4,4]
   + indicateurs[7,4,4] + indicateurs[8,4,4] + indicateurs[9,4,4] = 1
 R150: indicateurs[1,4,5] + indicateurs[2,4,5] + indicateurs[3,4,5]
   + indicateurs[4,4,5] + indicateurs[5,4,5] + indicateurs[6,4,5]
   + indicateurs[7,4,5] + indicateurs[8,4,5] + indicateurs[9,4,5] = 1
 R151: indicateurs[1,4,6] + indicateurs[2,4,6] + indicateurs[3,4,6]
   + indicateurs[4,4,6] + indicateurs[5,4,6] + indicateurs[6,4,6]
   + indicateurs[7,4,6] + indicateurs[8,4,6] + indicateurs[9,4,6] = 1
 R152: indicateurs[1,4,7] + indicateurs[2,4,7] + indicateurs[3,4,7]
   + indicateurs[4,4,7] + indicateurs[5,4,7] + indicateurs[6,4,7]
   + indicateurs[7,4,7] + indicateurs[8,4,7] + indicateurs[9,4,7] = 1
 R153: indicateurs[1,4,8] + indicateurs[2,4,8] + indicateurs[3,4,8]
   + indicateurs[4,4,8] + indicateurs[5,4,8] + indicateurs[6,4,8]
   + indicateurs[7,4,8] + indicateurs[8,4,8] + indicateurs[9,4,8] = 1
 R154: indicateurs[1,4,9] + indicateurs[2,4,9] + indicateurs[3,4,9]
   + indicateurs[4,4,9] + indicateurs[5,4,9] + indicateurs[6,4,9]
   + indicateurs[7,4,9] + indicateurs[8,4,9] + indicateurs[9,4,9] = 1
 R155: indicateurs[1,5,1] + indicateurs[2,5,1] + indicateurs[3,5,1]
   + indicateurs[4,5,1] + indicateurs[5,5,1] + indicateurs[6,5,1]
   + indicateurs[7,5,1] + indicateurs[8,5,1] + indicateurs[9,5,1] = 1
 R156: indicateurs[1,5,2] + indicateurs[2,5,2] + indicateurs[3,5,2]
   + indicateurs[4,5,2] + indicateurs[5,5,2] + indicateurs[6,5,2]
   + indicateurs[7,5,2] + indicateurs[8,5,2] + indicateurs[9,5,2] = 1
 R157: indicateurs[1,5,3] + indicateurs[2,5,3] + indicateurs[3,5,3]
   + indicateurs[4,5,3] + indicateurs[5,5,3] + indicateurs[6,5,3]
   + indicateurs[7,5,3] + indicateurs[8,5,3] + indicateurs[9,5,3] = 1
 R158: indicateurs[1,5,4] + indicateurs[2,5,4] + indicateurs[3,5,4]
   + indicateurs[4,5,4] + indicateurs[5,5,4] + indicateurs[6,5,4]
   + indicateurs[7,5,4] + indicateurs[8,5,4] + indicateurs[9,5,4] = 1
 R159: indicateurs[1,5,5] + indicateurs[2,5,5] + indicateurs[3,5,5]
   + indicateurs[4,5,5] + indicateurs[5,5,5] + indicateurs[6,5,5]
   + indicateurs[7,5,5] + indicateurs[8,5,5] + indicateurs[9,5,5] = 1
 R160: indicateurs[1,5,6] + indicateurs[2,5,6] + indicateurs[3,5,6]
   + indicateurs[4,5,6] + indicateurs[5,5,6] + indicateurs[6,5,6]
   + indicateurs[7,5,6] + indicateurs[8,5,6] + indicateurs[9,5,6] = 1
 R161: indicateurs[1,5,7] + indicateurs[2,5,7] + indicateurs[3,5,7]
   + indicateurs[4,5,7] + indicateurs[5,5,7] + indicateurs[6,5,7]
   + indicateurs[7,5,7] + indicateurs[8,5,7] + indicateurs[9,5,7] = 1
 R162: indicateurs[1,5,8] + indicateurs[2,5,8] + indicateurs[3,5,8]
   + indicateurs[4,5,8] + indicateurs[5,5,8] + indicateurs[6,5,8]
   + indicateurs[7,5,8] + indicateurs[8,5,8] + indicateurs[9,5,8] = 1
 R163: indicateurs[1,5,9] + indicateurs[2,5,9] + indicateurs[3,5,9]
   + indicateurs[4,5,9] + indicateurs[5,5,9] + indicateurs[6,5,9]
   + indicateurs[7,5,9] + indicateurs[8,5,9] + indicateurs[9,5,9] = 1
 R164: indicateurs[1,6,1] + indicateurs[2,6,1] + indicateurs[3,6,1]
   + indicateurs[4,6,1] + indicateurs[5,6,1] + indicateurs[6,6,1]
   + indicateurs[7,6,1] + indicateurs[8,6,1] + indicateurs[9,6,1] = 1
 R165: indicateurs[1,6,2] + indicateurs[2,6,2] + indicateurs[3,6,2]
   + indicateurs[4,6,2] + indicateurs[5,6,2] + indicateurs[6,6,2]
   + indicateurs[7,6,2] + indicateurs[8,6,2] + indicateurs[9,6,2] = 1
 R166: indicateurs[1,6,3] + indicateurs[2,6,3] + indicateurs[3,6,3]
   + indicateurs[4,6,3] + indicateurs[5,6,3] + indicateurs[6,6,3]
   + indicateurs[7,6,3] + indicateurs[8,6,3] + indicateurs[9,6,3] = 1
 R167: indicateurs[1,6,4] + indicateurs[2,6,4] + indicateurs[3,6,4]
   + indicateurs[4,6,4] + indicateurs[5,6,4] + indicateurs[6,6,4]
   + indicateurs[7,6,4] + indicateurs[8,6,4] + indicateurs[9,6,4] = 1
 R168: indicateurs[1,6,5] + indicateurs[2,6,5] + indicateurs[3,6,5]
   + indicateurs[4,6,5] + indicateurs[5,6,5] + indicateurs[6,6,5]
   + indicateurs[7,6,5] + indicateurs[8,6,5] + indicateurs[9,6,5] = 1
 R169: indicateurs[1,6,6] + indicateurs[2,6,6] + indicateurs[3,6,6]
   + indicateurs[4,6,6] + indicateurs[5,6,6] + indicateurs[6,6,6]
   + indicateurs[7,6,6] + indicateurs[8,6,6] + indicateurs[9,6,6] = 1
 R170: indicateurs[1,6,7] + indicateurs[2,6,7] + indicateurs[3,6,7]
   + indicateurs[4,6,7] + indicateurs[5,6,7] + indicateurs[6,6,7]
   + indicateurs[7,6,7] + indicateurs[8,6,7] + indicateurs[9,6,7] = 1
 R171: indicateurs[1,6,8] + indicateurs[2,6,8] + indicateurs[3,6,8]
   + indicateurs[4,6,8] + indicateurs[5,6,8] + indicateurs[6,6,8]
   + indicateurs[7,6,8] + indicateurs[8,6,8] + indicateurs[9,6,8] = 1
 R172: indicateurs[1,6,9] + indicateurs[2,6,9] + indicateurs[3,6,9]
   + indicateurs[4,6,9] + indicateurs[5,6,9] + indicateurs[6,6,9]
   + indicateurs[7,6,9] + indicateurs[8,6,9] + indicateurs[9,6,9] = 1
 R173: indicateurs[1,7,1] + indicateurs[2,7,1] + indicateurs[3,7,1]
   + indicateurs[4,7,1] + indicateurs[5,7,1] + indicateurs[6,7,1]
   + indicateurs[7,7,1] + indicateurs[8,7,1] + indicateurs[9,7,1] = 1
 R174: indicateurs[1,7,2] + indicateurs[2,7,2] + indicateurs[3,7,2]
   + indicateurs[4,7,2] + indicateurs[5,7,2] + indicateurs[6,7,2]
   + indicateurs[7,7,2] + indicateurs[8,7,2] + indicateurs[9,7,2] = 1
 R175: indicateurs[1,7,3] + indicateurs[2,7,3] + indicateurs[3,7,3]
   + indicateurs[4,7,3] + indicateurs[5,7,3] + indicateurs[6,7,3]
   + indicateurs[7,7,3] + indicateurs[8,7,3] + indicateurs[9,7,3] = 1
 R176: indicateurs[1,7,4] + indicateurs[2,7,4] + indicateurs[3,7,4]
   + indicateurs[4,7,4] + indicateurs[5,7,4] + indicateurs[6,7,4]
   + indicateurs[7,7,4] + indicateurs[8,7,4] + indicateurs[9,7,4] = 1
 R177: indicateurs[1,7,5] + indicateurs[2,7,5] + indicateurs[3,7,5]
   + indicateurs[4,7,5] + indicateurs[5,7,5] + indicateurs[6,7,5]
   + indicateurs[7,7,5] + indicateurs[8,7,5] + indicateurs[9,7,5] = 1
 R178: indicateurs[1,7,6] + indicateurs[2,7,6] + indicateurs[3,7,6]
   + indicateurs[4,7,6] + indicateurs[5,7,6] + indicateurs[6,7,6]
   + indicateurs[7,7,6] + indicateurs[8,7,6] + indicateurs[9,7,6] = 1
 R179: indicateurs[1,7,7] + indicateurs[2,7,7] + indicateurs[3,7,7]
   + indicateurs[4,7,7] + indicateurs[5,7,7] + indicateurs[6,7,7]
   + indicateurs[7,7,7] + indicateurs[8,7,7] + indicateurs[9,7,7] = 1
 R180: indicateurs[1,7,8] + indicateurs[2,7,8] + indicateurs[3,7,8]
   + indicateurs[4,7,8] + indicateurs[5,7,8] + indicateurs[6,7,8]
   + indicateurs[7,7,8] + indicateurs[8,7,8] + indicateurs[9,7,8] = 1
 R181: indicateurs[1,7,9] + indicateurs[2,7,9] + indicateurs[3,7,9]
   + indicateurs[4,7,9] + indicateurs[5,7,9] + indicateurs[6,7,9]
   + indicateurs[7,7,9] + indicateurs[8,7,9] + indicateurs[9,7,9] = 1
 R182: indicateurs[1,8,1] + indicateurs[2,8,1] + indicateurs[3,8,1]
   + indicateurs[4,8,1] + indicateurs[5,8,1] + indicateurs[6,8,1]
   + indicateurs[7,8,1] + indicateurs[8,8,1] + indicateurs[9,8,1] = 1
 R183: indicateurs[1,8,2] + indicateurs[2,8,2] + indicateurs[3,8,2]
   + indicateurs[4,8,2] + indicateurs[5,8,2] + indicateurs[6,8,2]
   + indicateurs[7,8,2] + indicateurs[8,8,2] + indicateurs[9,8,2] = 1
 R184: indicateurs[1,8,3] + indicateurs[2,8,3] + indicateurs[3,8,3]
   + indicateurs[4,8,3] + indicateurs[5,8,3] + indicateurs[6,8,3]
   + indicateurs[7,8,3] + indicateurs[8,8,3] + indicateurs[9,8,3] = 1
 R185: indicateurs[1,8,4] + indicateurs[2,8,4] + indicateurs[3,8,4]
   + indicateurs[4,8,4] + indicateurs[5,8,4] + indicateurs[6,8,4]
   + indicateurs[7,8,4] + indicateurs[8,8,4] + indicateurs[9,8,4] = 1
 R186: indicateurs[1,8,5] + indicateurs[2,8,5] + indicateurs[3,8,5]
   + indicateurs[4,8,5] + indicateurs[5,8,5] + indicateurs[6,8,5]
   + indicateurs[7,8,5] + indicateurs[8,8,5] + indicateurs[9,8,5] = 1
 R187: indicateurs[1,8,6] + indicateurs[2,8,6] + indicateurs[3,8,6]
   + indicateurs[4,8,6] + indicateurs[5,8,6] + indicateurs[6,8,6]
   + indicateurs[7,8,6] + indicateurs[8,8,6] + indicateurs[9,8,6] = 1
 R188: indicateurs[1,8,7] + indicateurs[2,8,7] + indicateurs[3,8,7]
   + indicateurs[4,8,7] + indicateurs[5,8,7] + indicateurs[6,8,7]
   + indicateurs[7,8,7] + indicateurs[8,8,7] + indicateurs[9,8,7] = 1
 R189: indicateurs[1,8,8] + indicateurs[2,8,8] + indicateurs[3,8,8]
   + indicateurs[4,8,8] + indicateurs[5,8,8] + indicateurs[6,8,8]
   + indicateurs[7,8,8] + indicateurs[8,8,8] + indicateurs[9,8,8] = 1
 R190: indicateurs[1,8,9] + indicateurs[2,8,9] + indicateurs[3,8,9]
   + indicateurs[4,8,9] + indicateurs[5,8,9] + indicateurs[6,8,9]
   + indicateurs[7,8,9] + indicateurs[8,8,9] + indicateurs[9,8,9] = 1
 R191: indicateurs[1,9,1] + indicateurs[2,9,1] + indicateurs[3,9,1]
   + indicateurs[4,9,1] + indicateurs[5,9,1] + indicateurs[6,9,1]
   + indicateurs[7,9,1] + indicateurs[8,9,1] + indicateurs[9,9,1] = 1
 R192: indicateurs[1,9,2] + indicateurs[2,9,2] + indicateurs[3,9,2]
   + indicateurs[4,9,2] + indicateurs[5,9,2] + indicateurs[6,9,2]
   + indicateurs[7,9,2] + indicateurs[8,9,2] + indicateurs[9,9,2] = 1
 R193: indicateurs[1,9,3] + indicateurs[2,9,3] + indicateurs[3,9,3]
   + indicateurs[4,9,3] + indicateurs[5,9,3] + indicateurs[6,9,3]
   + indicateurs[7,9,3] + indicateurs[8,9,3] + indicateurs[9,9,3] = 1
 R194: indicateurs[1,9,4] + indicateurs[2,9,4] + indicateurs[3,9,4]
   + indicateurs[4,9,4] + indicateurs[5,9,4] + indicateurs[6,9,4]
   + indicateurs[7,9,4] + indicateurs[8,9,4] + indicateurs[9,9,4] = 1
 R195: indicateurs[1,9,5] + indicateurs[2,9,5] + indicateurs[3,9,5]
   + indicateurs[4,9,5] + indicateurs[5,9,5] + indicateurs[6,9,5]
   + indicateurs[7,9,5] + indicateurs[8,9,5] + indicateurs[9,9,5] = 1
 R196: indicateurs[1,9,6] + indicateurs[2,9,6] + indicateurs[3,9,6]
   + indicateurs[4,9,6] + indicateurs[5,9,6] + indicateurs[6,9,6]
   + indicateurs[7,9,6] + indicateurs[8,9,6] + indicateurs[9,9,6] = 1
 R197: indicateurs[1,9,7] + indicateurs[2,9,7] + indicateurs[3,9,7]
   + indicateurs[4,9,7] + indicateurs[5,9,7] + indicateurs[6,9,7]
   + indicateurs[7,9,7] + indicateurs[8,9,7] + indicateurs[9,9,7] = 1
 R198: indicateurs[1,9,8] + indicateurs[2,9,8] + indicateurs[3,9,8]
   + indicateurs[4,9,8] + indicateurs[5,9,8] + indicateurs[6,9,8]
   + indicateurs[7,9,8] + indicateurs[8,9,8] + indicateurs[9,9,8] = 1
 R199: indicateurs[1,9,9] + indicateurs[2,9,9] + indicateurs[3,9,9]
   + indicateurs[4,9,9] + indicateurs[5,9,9] + indicateurs[6,9,9]
   + indicateurs[7,9,9] + indicateurs[8,9,9] + indicateurs[9,9,9] = 1
 R200: indicateurs[1,1,1] + indicateurs[1,2,1] + indicateurs[1,3,1]
   + indicateurs[2,1,1] + indicateurs[2,2,1] + indicateurs[2,3,1]
   + indicateurs[3,1,1] + indicateurs[3,2,1] + indicateurs[3,3,1] = 1
 R201: indicateurs[1,4,1] + indicateurs[1,5,1] + indicateurs[1,6,1]
   + indicateurs[2,4,1] + indicateurs[2,5,1] + indicateurs[2,6,1]
   + indicateurs[3,4,1] + indicateurs[3,5,1] + indicateurs[3,6,1] = 1
 R202: indicateurs[1,7,1] + indicateurs[1,8,1] + indicateurs[1,9,1]
   + indicateurs[2,7,1] + indicateurs[2,8,1] + indicateurs[2,9,1]
   + indicateurs[3,7,1] + indicateurs[3,8,1] + indicateurs[3,9,1] = 1
 R203: indicateurs[4,1,1] + indicateurs[4,2,1] + indicateurs[4,3,1]
   + indicateurs[5,1,1] + indicateurs[5,2,1] + indicateurs[5,3,1]
   + indicateurs[6,1,1] + indicateurs[6,2,1] + indicateurs[6,3,1] = 1
 R204: indicateurs[4,4,1] + indicateurs[4,5,1] + indicateurs[4,6,1]
   + indicateurs[5,4,1] + indicateurs[5,5,1] + indicateurs[5,6,1]
   + indicateurs[6,4,1] + indicateurs[6,5,1] + indicateurs[6,6,1] = 1
 R205: indicateurs[4,7,1] + indicateurs[4,8,1] + indicateurs[4,9,1]
   + indicateurs[5,7,1] + indicateurs[5,8,1] + indicateurs[5,9,1]
   + indicateurs[6,7,1] + indicateurs[6,8,1] + indicateurs[6,9,1] = 1
 R206: indicateurs[7,1,1] + indicateurs[7,2,1] + indicateurs[7,3,1]
   + indicateurs[8,1,1] + indicateurs[8,2,1] + indicateurs[8,3,1]
   + indicateurs[9,1,1] + indicateurs[9,2,1] + indicateurs[9,3,1] = 1
 R207: indicateurs[7,4,1] + indicateurs[7,5,1] + indicateurs[7,6,1]
   + indicateurs[8,4,1] + indicateurs[8,5,1] + indicateurs[8,6,1]
   + indicateurs[9,4,1] + indicateurs[9,5,1] + indicateurs[9,6,1] = 1
 R208: indicateurs[7,7,1] + indicateurs[7,8,1] + indicateurs[7,9,1]
   + indicateurs[8,7,1] + indicateurs[8,8,1] + indicateurs[8,9,1]
   + indicateurs[9,7,1] + indicateurs[9,8,1] + indicateurs[9,9,1] = 1
 R209: indicateurs[1,1,2] + indicateurs[1,2,2] + indicateurs[1,3,2]
   + indicateurs[2,1,2] + indicateurs[2,2,2] + indicateurs[2,3,2]
   + indicateurs[3,1,2] + indicateurs[3,2,2] + indicateurs[3,3,2] = 1
 R210: indicateurs[1,4,2] + indicateurs[1,5,2] + indicateurs[1,6,2]
   + indicateurs[2,4,2] + indicateurs[2,5,2] + indicateurs[2,6,2]
   + indicateurs[3,4,2] + indicateurs[3,5,2] + indicateurs[3,6,2] = 1
 R211: indicateurs[1,7,2] + indicateurs[1,8,2] + indicateurs[1,9,2]
   + indicateurs[2,7,2] + indicateurs[2,8,2] + indicateurs[2,9,2]
   + indicateurs[3,7,2] + indicateurs[3,8,2] + indicateurs[3,9,2] = 1
 R212: indicateurs[4,1,2] + indicateurs[4,2,2] + indicateurs[4,3,2]
   + indicateurs[5,1,2] + indicateurs[5,2,2] + indicateurs[5,3,2]
   + indicateurs[6,1,2] + indicateurs[6,2,2] + indicateurs[6,3,2] = 1
 R213: indicateurs[4,4,2] + indicateurs[4,5,2] + indicateurs[4,6,2]
   + indicateurs[5,4,2] + indicateurs[5,5,2] + indicateurs[5,6,2]
   + indicateurs[6,4,2] + indicateurs[6,5,2] + indicateurs[6,6,2] = 1
 R214: indicateurs[4,7,2] + indicateurs[4,8,2] + indicateurs[4,9,2]
   + indicateurs[5,7,2] + indicateurs[5,8,2] + indicateurs[5,9,2]
   + indicateurs[6,7,2] + indicateurs[6,8,2] + indicateurs[6,9,2] = 1
 R215: indicateurs[7,1,2] + indicateurs[7,2,2] + indicateurs[7,3,2]
   + indicateurs[8,1,2] + indicateurs[8,2,2] + indicateurs[8,3,2]
   + indicateurs[9,1,2] + indicateurs[9,2,2] + indicateurs[9,3,2] = 1
 R216: indicateurs[7,4,2] + indicateurs[7,5,2] + indicateurs[7,6,2]
   + indicateurs[8,4,2] + indicateurs[8,5,2] + indicateurs[8,6,2]
   + indicateurs[9,4,2] + indicateurs[9,5,2] + indicateurs[9,6,2] = 1
 R217: indicateurs[7,7,2] + indicateurs[7,8,2] + indicateurs[7,9,2]
   + indicateurs[8,7,2] + indicateurs[8,8,2] + indicateurs[8,9,2]
   + indicateurs[9,7,2] + indicateurs[9,8,2] + indicateurs[9,9,2] = 1
 R218: indicateurs[1,1,3] + indicateurs[1,2,3] + indicateurs[1,3,3]
   + indicateurs[2,1,3] + indicateurs[2,2,3] + indicateurs[2,3,3]
   + indicateurs[3,1,3] + indicateurs[3,2,3] + indicateurs[3,3,3] = 1
 R219: indicateurs[1,4,3] + indicateurs[1,5,3] + indicateurs[1,6,3]
   + indicateurs[2,4,3] + indicateurs[2,5,3] + indicateurs[2,6,3]
   + indicateurs[3,4,3] + indicateurs[3,5,3] + indicateurs[3,6,3] = 1
 R220: indicateurs[1,7,3] + indicateurs[1,8,3] + indicateurs[1,9,3]
   + indicateurs[2,7,3] + indicateurs[2,8,3] + indicateurs[2,9,3]
   + indicateurs[3,7,3] + indicateurs[3,8,3] + indicateurs[3,9,3] = 1
 R221: indicateurs[4,1,3] + indicateurs[4,2,3] + indicateurs[4,3,3]
   + indicateurs[5,1,3] + indicateurs[5,2,3] + indicateurs[5,3,3]
   + indicateurs[6,1,3] + indicateurs[6,2,3] + indicateurs[6,3,3] = 1
 R222: indicateurs[4,4,3] + indicateurs[4,5,3] + indicateurs[4,6,3]
   + indicateurs[5,4,3] + indicateurs[5,5,3] + indicateurs[5,6,3]
   + indicateurs[6,4,3] + indicateurs[6,5,3] + indicateurs[6,6,3] = 1
 R223: indicateurs[4,7,3] + indicateurs[4,8,3] + indicateurs[4,9,3]
   + indicateurs[5,7,3] + indicateurs[5,8,3] + indicateurs[5,9,3]
   + indicateurs[6,7,3] + indicateurs[6,8,3] + indicateurs[6,9,3] = 1
 R224: indicateurs[7,1,3] + indicateurs[7,2,3] + indicateurs[7,3,3]
   + indicateurs[8,1,3] + indicateurs[8,2,3] + indicateurs[8,3,3]
   + indicateurs[9,1,3] + indicateurs[9,2,3] + indicateurs[9,3,3] = 1
 R225: indicateurs[7,4,3] + indicateurs[7,5,3] + indicateurs[7,6,3]
   + indicateurs[8,4,3] + indicateurs[8,5,3] + indicateurs[8,6,3]
   + indicateurs[9,4,3] + indicateurs[9,5,3] + indicateurs[9,6,3] = 1
 R226: indicateurs[7,7,3] + indicateurs[7,8,3] + indicateurs[7,9,3]
   + indicateurs[8,7,3] + indicateurs[8,8,3] + indicateurs[8,9,3]
   + indicateurs[9,7,3] + indicateurs[9,8,3] + indicateurs[9,9,3] = 1
 R227: indicateurs[1,1,4] + indicateurs[1,2,4] + indicateurs[1,3,4]
   + indicateurs[2,1,4] + indicateurs[2,2,4] + indicateurs[2,3,4]
   + indicateurs[3,1,4] + indicateurs[3,2,4] + indicateurs[3,3,4] = 1
 R228: indicateurs[1,4,4] + indicateurs[1,5,4] + indicateurs[1,6,4]
   + indicateurs[2,4,4] + indicateurs[2,5,4] + indicateurs[2,6,4]
   + indicateurs[3,4,4] + indicateurs[3,5,4] + indicateurs[3,6,4] = 1
 R229: indicateurs[1,7,4] + indicateurs[1,8,4] + indicateurs[1,9,4]
   + indicateurs[2,7,4] + indicateurs[2,8,4] + indicateurs[2,9,4]
   + indicateurs[3,7,4] + indicateurs[3,8,4] + indicateurs[3,9,4] = 1
 R230: indicateurs[4,1,4] + indicateurs[4,2,4] + indicateurs[4,3,4]
   + indicateurs[5,1,4] + indicateurs[5,2,4] + indicateurs[5,3,4]
   + indicateurs[6,1,4] + indicateurs[6,2,4] + indicateurs[6,3,4] = 1
 R231: indicateurs[4,4,4] + indicateurs[4,5,4] + indicateurs[4,6,4]
   + indicateurs[5,4,4] + indicateurs[5,5,4] + indicateurs[5,6,4]
   + indicateurs[6,4,4] + indicateurs[6,5,4] + indicateurs[6,6,4] = 1
 R232: indicateurs[4,7,4] + indicateurs[4,8,4] + indicateurs[4,9,4]
   + indicateurs[5,7,4] + indicateurs[5,8,4] + indicateurs[5,9,4]
   + indicateurs[6,7,4] + indicateurs[6,8,4] + indicateurs[6,9,4] = 1
 R233: indicateurs[7,1,4] + indicateurs[7,2,4] + indicateurs[7,3,4]
   + indicateurs[8,1,4] + indicateurs[8,2,4] + indicateurs[8,3,4]
   + indicateurs[9,1,4] + indicateurs[9,2,4] + indicateurs[9,3,4] = 1
 R234: indicateurs[7,4,4] + indicateurs[7,5,4] + indicateurs[7,6,4]
   + indicateurs[8,4,4] + indicateurs[8,5,4] + indicateurs[8,6,4]
   + indicateurs[9,4,4] + indicateurs[9,5,4] + indicateurs[9,6,4] = 1
 R235: indicateurs[7,7,4] + indicateurs[7,8,4] + indicateurs[7,9,4]
   + indicateurs[8,7,4] + indicateurs[8,8,4] + indicateurs[8,9,4]
   + indicateurs[9,7,4] + indicateurs[9,8,4] + indicateurs[9,9,4] = 1
 R236: indicateurs[1,1,5] + indicateurs[1,2,5] + indicateurs[1,3,5]
   + indicateurs[2,1,5] + indicateurs[2,2,5] + indicateurs[2,3,5]
   + indicateurs[3,1,5] + indicateurs[3,2,5] + indicateurs[3,3,5] = 1
 R237: indicateurs[1,4,5] + indicateurs[1,5,5] + indicateurs[1,6,5]
   + indicateurs[2,4,5] + indicateurs[2,5,5] + indicateurs[2,6,5]
   + indicateurs[3,4,5] + indicateurs[3,5,5] + indicateurs[3,6,5] = 1
 R238: indicateurs[1,7,5] + indicateurs[1,8,5] + indicateurs[1,9,5]
   + indicateurs[2,7,5] + indicateurs[2,8,5] + indicateurs[2,9,5]
   + indicateurs[3,7,5] + indicateurs[3,8,5] + indicateurs[3,9,5] = 1
 R239: indicateurs[4,1,5] + indicateurs[4,2,5] + indicateurs[4,3,5]
   + indicateurs[5,1,5] + indicateurs[5,2,5] + indicateurs[5,3,5]
   + indicateurs[6,1,5] + indicateurs[6,2,5] + indicateurs[6,3,5] = 1
 R240: indicateurs[4,4,5] + indicateurs[4,5,5] + indicateurs[4,6,5]
   + indicateurs[5,4,5] + indicateurs[5,5,5] + indicateurs[5,6,5]
   + indicateurs[6,4,5] + indicateurs[6,5,5] + indicateurs[6,6,5] = 1
 R241: indicateurs[4,7,5] + indicateurs[4,8,5] + indicateurs[4,9,5]
   + indicateurs[5,7,5] + indicateurs[5,8,5] + indicateurs[5,9,5]
   + indicateurs[6,7,5] + indicateurs[6,8,5] + indicateurs[6,9,5] = 1
 R242: indicateurs[7,1,5] + indicateurs[7,2,5] + indicateurs[7,3,5]
   + indicateurs[8,1,5] + indicateurs[8,2,5] + indicateurs[8,3,5]
   + indicateurs[9,1,5] + indicateurs[9,2,5] + indicateurs[9,3,5] = 1
 R243: indicateurs[7,4,5] + indicateurs[7,5,5] + indicateurs[7,6,5]
   + indicateurs[8,4,5] + indicateurs[8,5,5] + indicateurs[8,6,5]
   + indicateurs[9,4,5] + indicateurs[9,5,5] + indicateurs[9,6,5] = 1
 R244: indicateurs[7,7,5] + indicateurs[7,8,5] + indicateurs[7,9,5]
   + indicateurs[8,7,5] + indicateurs[8,8,5] + indicateurs[8,9,5]
   + indicateurs[9,7,5] + indicateurs[9,8,5] + indicateurs[9,9,5] = 1
 R245: indicateurs[1,1,6] + indicateurs[1,2,6] + indicateurs[1,3,6]
   + indicateurs[2,1,6] + indicateurs[2,2,6] + indicateurs[2,3,6]
   + indicateurs[3,1,6] + indicateurs[3,2,6] + indicateurs[3,3,6] = 1
 R246: indicateurs[1,4,6] + indicateurs[1,5,6] + indicateurs[1,6,6]
   + indicateurs[2,4,6] + indicateurs[2,5,6] + indicateurs[2,6,6]
   + indicateurs[3,4,6] + indicateurs[3,5,6] + indicateurs[3,6,6] = 1
 R247: indicateurs[1,7,6] + indicateurs[1,8,6] + indicateurs[1,9,6]
   + indicateurs[2,7,6] + indicateurs[2,8,6] + indicateurs[2,9,6]
   + indicateurs[3,7,6] + indicateurs[3,8,6] + indicateurs[3,9,6] = 1
 R248: indicateurs[4,1,6] + indicateurs[4,2,6] + indicateurs[4,3,6]
   + indicateurs[5,1,6] + indicateurs[5,2,6] + indicateurs[5,3,6]
   + indicateurs[6,1,6] + indicateurs[6,2,6] + indicateurs[6,3,6] = 1
 R249: indicateurs[4,4,6] + indicateurs[4,5,6] + indicateurs[4,6,6]
   + indicateurs[5,4,6] + indicateurs[5,5,6] + indicateurs[5,6,6]
   + indicateurs[6,4,6] + indicateurs[6,5,6] + indicateurs[6,6,6] = 1
 R250: indicateurs[4,7,6] + indicateurs[4,8,6] + indicateurs[4,9,6]
   + indicateurs[5,7,6] + indicateurs[5,8,6] + indicateurs[5,9,6]
   + indicateurs[6,7,6] + indicateurs[6,8,6] + indicateurs[6,9,6] = 1
 R251: indicateurs[7,1,6] + indicateurs[7,2,6] + indicateurs[7,3,6]
   + indicateurs[8,1,6] + indicateurs[8,2,6] + indicateurs[8,3,6]
   + indicateurs[9,1,6] + indicateurs[9,2,6] + indicateurs[9,3,6] = 1
 R252: indicateurs[7,4,6] + indicateurs[7,5,6] + indicateurs[7,6,6]
   + indicateurs[8,4,6] + indicateurs[8,5,6] + indicateurs[8,6,6]
   + indicateurs[9,4,6] + indicateurs[9,5,6] + indicateurs[9,6,6] = 1
 R253: indicateurs[7,7,6] + indicateurs[7,8,6] + indicateurs[7,9,6]
   + indicateurs[8,7,6] + indicateurs[8,8,6] + indicateurs[8,9,6]
   + indicateurs[9,7,6] + indicateurs[9,8,6] + indicateurs[9,9,6] = 1
 R254: indicateurs[1,1,7] + indicateurs[1,2,7] + indicateurs[1,3,7]
   + indicateurs[2,1,7] + indicateurs[2,2,7] + indicateurs[2,3,7]
   + indicateurs[3,1,7] + indicateurs[3,2,7] + indicateurs[3,3,7] = 1
 R255: indicateurs[1,4,7] + indicateurs[1,5,7] + indicateurs[1,6,7]
   + indicateurs[2,4,7] + indicateurs[2,5,7] + indicateurs[2,6,7]
   + indicateurs[3,4,7] + indicateurs[3,5,7] + indicateurs[3,6,7] = 1
 R256: indicateurs[1,7,7] + indicateurs[1,8,7] + indicateurs[1,9,7]
   + indicateurs[2,7,7] + indicateurs[2,8,7] + indicateurs[2,9,7]
   + indicateurs[3,7,7] + indicateurs[3,8,7] + indicateurs[3,9,7] = 1
 R257: indicateurs[4,1,7] + indicateurs[4,2,7] + indicateurs[4,3,7]
   + indicateurs[5,1,7] + indicateurs[5,2,7] + indicateurs[5,3,7]
   + indicateurs[6,1,7] + indicateurs[6,2,7] + indicateurs[6,3,7] = 1
 R258: indicateurs[4,4,7] + indicateurs[4,5,7] + indicateurs[4,6,7]
   + indicateurs[5,4,7] + indicateurs[5,5,7] + indicateurs[5,6,7]
   + indicateurs[6,4,7] + indicateurs[6,5,7] + indicateurs[6,6,7] = 1
 R259: indicateurs[4,7,7] + indicateurs[4,8,7] + indicateurs[4,9,7]
   + indicateurs[5,7,7] + indicateurs[5,8,7] + indicateurs[5,9,7]
   + indicateurs[6,7,7] + indicateurs[6,8,7] + indicateurs[6,9,7] = 1
 R260: indicateurs[7,1,7] + indicateurs[7,2,7] + indicateurs[7,3,7]
   + indicateurs[8,1,7] + indicateurs[8,2,7] + indicateurs[8,3,7]
   + indicateurs[9,1,7] + indicateurs[9,2,7] + indicateurs[9,3,7] = 1
 R261: indicateurs[7,4,7] + indicateurs[7,5,7] + indicateurs[7,6,7]
   + indicateurs[8,4,7] + indicateurs[8,5,7] + indicateurs[8,6,7]
   + indicateurs[9,4,7] + indicateurs[9,5,7] + indicateurs[9,6,7] = 1
 R262: indicateurs[7,7,7] + indicateurs[7,8,7] + indicateurs[7,9,7]
   + indicateurs[8,7,7] + indicateurs[8,8,7] + indicateurs[8,9,7]
   + indicateurs[9,7,7] + indicateurs[9,8,7] + indicateurs[9,9,7] = 1
 R263: indicateurs[1,1,8] + indicateurs[1,2,8] + indicateurs[1,3,8]
   + indicateurs[2,1,8] + indicateurs[2,2,8] + indicateurs[2,3,8]
   + indicateurs[3,1,8] + indicateurs[3,2,8] + indicateurs[3,3,8] = 1
 R264: indicateurs[1,4,8] + indicateurs[1,5,8] + indicateurs[1,6,8]
   + indicateurs[2,4,8] + indicateurs[2,5,8] + indicateurs[2,6,8]
   + indicateurs[3,4,8] + indicateurs[3,5,8] + indicateurs[3,6,8] = 1
 R265: indicateurs[1,7,8] + indicateurs[1,8,8] + indicateurs[1,9,8]
   + indicateurs[2,7,8] + indicateurs[2,8,8] + indicateurs[2,9,8]
   + indicateurs[3,7,8] + indicateurs[3,8,8] + indicateurs[3,9,8] = 1
 R266: indicateurs[4,1,8] + indicateurs[4,2,8] + indicateurs[4,3,8]
   + indicateurs[5,1,8] + indicateurs[5,2,8] + indicateurs[5,3,8]
   + indicateurs[6,1,8] + indicateurs[6,2,8] + indicateurs[6,3,8] = 1
 R267: indicateurs[4,4,8] + indicateurs[4,5,8] + indicateurs[4,6,8]
   + indicateurs[5,4,8] + indicateurs[5,5,8] + indicateurs[5,6,8]
   + indicateurs[6,4,8] + indicateurs[6,5,8] + indicateurs[6,6,8] = 1
 R268: indicateurs[4,7,8] + indicateurs[4,8,8] + indicateurs[4,9,8]
   + indicateurs[5,7,8] + indicateurs[5,8,8] + indicateurs[5,9,8]
   + indicateurs[6,7,8] + indicateurs[6,8,8] + indicateurs[6,9,8] = 1
 R269: indicateurs[7,1,8] + indicateurs[7,2,8] + indicateurs[7,3,8]
   + indicateurs[8,1,8] + indicateurs[8,2,8] + indicateurs[8,3,8]
   + indicateurs[9,1,8] + indicateurs[9,2,8] + indicateurs[9,3,8] = 1
 R270: indicateurs[7,4,8] + indicateurs[7,5,8] + indicateurs[7,6,8]
   + indicateurs[8,4,8] + indicateurs[8,5,8] + indicateurs[8,6,8]
   + indicateurs[9,4,8] + indicateurs[9,5,8] + indicateurs[9,6,8] = 1
 R271: indicateurs[7,7,8] + indicateurs[7,8,8] + indicateurs[7,9,8]
   + indicateurs[8,7,8] + indicateurs[8,8,8] + indicateurs[8,9,8]
   + indicateurs[9,7,8] + indicateurs[9,8,8] + indicateurs[9,9,8] = 1
 R272: indicateurs[1,1,9] + indicateurs[1,2,9] + indicateurs[1,3,9]
   + indicateurs[2,1,9] + indicateurs[2,2,9] + indicateurs[2,3,9]
   + indicateurs[3,1,9] + indicateurs[3,2,9] + indicateurs[3,3,9] = 1
 R273: indicateurs[1,4,9] + indicateurs[1,5,9] + indicateurs[1,6,9]
   + indicateurs[2,4,9] + indicateurs[2,5,9] + indicateurs[2,6,9]
   + indicateurs[3,4,9] + indicateurs[3,5,9] + indicateurs[3,6,9] = 1
 R274: indicateurs[1,7,9] + indicateurs[1,8,9] + indicateurs[1,9,9]
   + indicateurs[2,7,9] + indicateurs[2,8,9] + indicateurs[2,9,9]
   + indicateurs[3,7,9] + indicateurs[3,8,9] + indicateurs[3,9,9] = 1
 R275: indicateurs[4,1,9] + indicateurs[4,2,9] + indicateurs[4,3,9]
   + indicateurs[5,1,9] + indicateurs[5,2,9] + indicateurs[5,3,9]
   + indicateurs[6,1,9] + indicateurs[6,2,9] + indicateurs[6,3,9] = 1
 R276: indicateurs[4,4,9] + indicateurs[4,5,9] + indicateurs[4,6,9]
   + indicateurs[5,4,9] + indicateurs[5,5,9] + indicateurs[5,6,9]
   + indicateurs[6,4,9] + indicateurs[6,5,9] + indicateurs[6,6,9] = 1
 R277: indicateurs[4,7,9] + indicateurs[4,8,9] + indicateurs[4,9,9]
   + indicateurs[5,7,9] + indicateurs[5,8,9] + indicateurs[5,9,9]
   + indicateurs[6,7,9] + indicateurs[6,8,9] + indicateurs[6,9,9] = 1
 R278: indicateurs[7,1,9] + indicateurs[7,2,9] + indicateurs[7,3,9]
   + indicateurs[8,1,9] + indicateurs[8,2,9] + indicateurs[8,3,9]
   + indicateurs[9,1,9] + indicateurs[9,2,9] + indicateurs[9,3,9] = 1
 R279: indicateurs[7,4,9] + indicateurs[7,5,9] + indicateurs[7,6,9]
   + indicateurs[8,4,9] + indicateurs[8,5,9] + indicateurs[8,6,9]
   + indicateurs[9,4,9] + indicateurs[9,5,9] + indicateurs[9,6,9] = 1
 R280: indicateurs[7,7,9] + indicateurs[7,8,9] + indicateurs[7,9,9]
   + indicateurs[8,7,9] + indicateurs[8,8,9] + indicateurs[8,9,9]
   + indicateurs[9,7,9] + indicateurs[9,8,9] + indicateurs[9,9,9] = 1
Bounds
Binaries
 indicateurs[1,1,1] indicateurs[1,1,2] indicateurs[1,1,3]
 indicateurs[1,1,4] indicateurs[1,1,5] indicateurs[1,1,6]
 indicateurs[1,1,7] indicateurs[1,1,8] indicateurs[1,1,9]
 indicateurs[1,2,1] indicateurs[1,2,2] indicateurs[1,2,3]
 indicateurs[1,2,4] indicateurs[1,2,5] indicateurs[1,2,6]
 indicateurs[1,2,7] indicateurs[1,2,8] indicateurs[1,2,9]
 indicateurs[1,3,1] indicateurs[1,3,2] indicateurs[1,3,3]
 indicateurs[1,3,4] indicateurs[1,3,5] indicateurs[1,3,6]
 indicateurs[1,3,7] indicateurs[1,3,8] indicateurs[1,3,9]
 indicateurs[1,4,1] indicateurs[1,4,2] indicateurs[1,4,3]
 indicateurs[1,4,4] indicateurs[1,4,5] indicateurs[1,4,6]
 indicateurs[1,4,7] indicateurs[1,4,8] indicateurs[1,4,9]
 indicateurs[1,5,1] indicateurs[1,5,2] indicateurs[1,5,3]
 indicateurs[1,5,4] indicateurs[1,5,5] indicateurs[1,5,6]
 indicateurs[1,5,7] indicateurs[1,5,8] indicateurs[1,5,9]
 indicateurs[1,6,1] indicateurs[1,6,2] indicateurs[1,6,3]
 indicateurs[1,6,4] indicateurs[1,6,5] indicateurs[1,6,6]
 indicateurs[1,6,7] indicateurs[1,6,8] indicateurs[1,6,9]
 indicateurs[1,7,1] indicateurs[1,7,2] indicateurs[1,7,3]
 indicateurs[1,7,4] indicateurs[1,7,5] indicateurs[1,7,6]
 indicateurs[1,7,7] indicateurs[1,7,8] indicateurs[1,7,9]
 indicateurs[1,8,1] indicateurs[1,8,2] indicateurs[1,8,3]
 indicateurs[1,8,4] indicateurs[1,8,5] indicateurs[1,8,6]
 indicateurs[1,8,7] indicateurs[1,8,8] indicateurs[1,8,9]
 indicateurs[1,9,1] indicateurs[1,9,2] indicateurs[1,9,3]
 indicateurs[1,9,4] indicateurs[1,9,5] indicateurs[1,9,6]
 indicateurs[1,9,7] indicateurs[1,9,8] indicateurs[1,9,9]
 indicateurs[2,1,1] indicateurs[2,1,2] indicateurs[2,1,3]
 indicateurs[2,1,4] indicateurs[2,1,5] indicateurs[2,1,6]
 indicateurs[2,1,7] indicateurs[2,1,8] indicateurs[2,1,9]
 indicateurs[2,2,1] indicateurs[2,2,2] indicateurs[2,2,3]
 indicateurs[2,2,4] indicateurs[2,2,5] indicateurs[2,2,6]
 indicateurs[2,2,7] indicateurs[2,2,8] indicateurs[2,2,9]
 indicateurs[2,3,1] indicateurs[2,3,2] indicateurs[2,3,3]
 indicateurs[2,3,4] indicateurs[2,3,5] indicateurs[2,3,6]
 indicateurs[2,3,7] indicateurs[2,3,8] indicateurs[2,3,9]
 indicateurs[2,4,1] indicateurs[2,4,2] indicateurs[2,4,3]
 indicateurs[2,4,4] indicateurs[2,4,5] indicateurs[2,4,6]
 indicateurs[2,4,7] indicateurs[2,4,8] indicateurs[2,4,9]
 indicateurs[2,5,1] indicateurs[2,5,2] indicateurs[2,5,3]
 indicateurs[2,5,4] indicateurs[2,5,5] indicateurs[2,5,6]
 indicateurs[2,5,7] indicateurs[2,5,8] indicateurs[2,5,9]
 indicateurs[2,6,1] indicateurs[2,6,2] indicateurs[2,6,3]
 indicateurs[2,6,4] indicateurs[2,6,5] indicateurs[2,6,6]
 indicateurs[2,6,7] indicateurs[2,6,8] indicateurs[2,6,9]
 indicateurs[2,7,1] indicateurs[2,7,2] indicateurs[2,7,3]
 indicateurs[2,7,4] indicateurs[2,7,5] indicateurs[2,7,6]
 indicateurs[2,7,7] indicateurs[2,7,8] indicateurs[2,7,9]
 indicateurs[2,8,1] indicateurs[2,8,2] indicateurs[2,8,3]
 indicateurs[2,8,4] indicateurs[2,8,5] indicateurs[2,8,6]
 indicateurs[2,8,7] indicateurs[2,8,8] indicateurs[2,8,9]
 indicateurs[2,9,1] indicateurs[2,9,2] indicateurs[2,9,3]
 indicateurs[2,9,4] indicateurs[2,9,5] indicateurs[2,9,6]
 indicateurs[2,9,7] indicateurs[2,9,8] indicateurs[2,9,9]
 indicateurs[3,1,1] indicateurs[3,1,2] indicateurs[3,1,3]
 indicateurs[3,1,4] indicateurs[3,1,5] indicateurs[3,1,6]
 indicateurs[3,1,7] indicateurs[3,1,8] indicateurs[3,1,9]
 indicateurs[3,2,1] indicateurs[3,2,2] indicateurs[3,2,3]
 indicateurs[3,2,4] indicateurs[3,2,5] indicateurs[3,2,6]
 indicateurs[3,2,7] indicateurs[3,2,8] indicateurs[3,2,9]
 indicateurs[3,3,1] indicateurs[3,3,2] indicateurs[3,3,3]
 indicateurs[3,3,4] indicateurs[3,3,5] indicateurs[3,3,6]
 indicateurs[3,3,7] indicateurs[3,3,8] indicateurs[3,3,9]
 indicateurs[3,4,1] indicateurs[3,4,2] indicateurs[3,4,3]
 indicateurs[3,4,4] indicateurs[3,4,5] indicateurs[3,4,6]
 indicateurs[3,4,7] indicateurs[3,4,8] indicateurs[3,4,9]
 indicateurs[3,5,1] indicateurs[3,5,2] indicateurs[3,5,3]
 indicateurs[3,5,4] indicateurs[3,5,5] indicateurs[3,5,6]
 indicateurs[3,5,7] indicateurs[3,5,8] indicateurs[3,5,9]
 indicateurs[3,6,1] indicateurs[3,6,2] indicateurs[3,6,3]
 indicateurs[3,6,4] indicateurs[3,6,5] indicateurs[3,6,6]
 indicateurs[3,6,7] indicateurs[3,6,8] indicateurs[3,6,9]
 indicateurs[3,7,1] indicateurs[3,7,2] indicateurs[3,7,3]
 indicateurs[3,7,4] indicateurs[3,7,5] indicateurs[3,7,6]
 indicateurs[3,7,7] indicateurs[3,7,8] indicateurs[3,7,9]
 indicateurs[3,8,1] indicateurs[3,8,2] indicateurs[3,8,3]
 indicateurs[3,8,4] indicateurs[3,8,5] indicateurs[3,8,6]
 indicateurs[3,8,7] indicateurs[3,8,8] indicateurs[3,8,9]
 indicateurs[3,9,1] indicateurs[3,9,2] indicateurs[3,9,3]
 indicateurs[3,9,4] indicateurs[3,9,5] indicateurs[3,9,6]
 indicateurs[3,9,7] indicateurs[3,9,8] indicateurs[3,9,9]
 indicateurs[4,1,1] indicateurs[4,1,2] indicateurs[4,1,3]
 indicateurs[4,1,4] indicateurs[4,1,5] indicateurs[4,1,6]
 indicateurs[4,1,7] indicateurs[4,1,8] indicateurs[4,1,9]
 indicateurs[4,2,1] indicateurs[4,2,2] indicateurs[4,2,3]
 indicateurs[4,2,4] indicateurs[4,2,5] indicateurs[4,2,6]
 indicateurs[4,2,7] indicateurs[4,2,8] indicateurs[4,2,9]
 indicateurs[4,3,1] indicateurs[4,3,2] indicateurs[4,3,3]
 indicateurs[4,3,4] indicateurs[4,3,5] indicateurs[4,3,6]
 indicateurs[4,3,7] indicateurs[4,3,8] indicateurs[4,3,9]
 indicateurs[4,4,1] indicateurs[4,4,2] indicateurs[4,4,3]
 indicateurs[4,4,4] indicateurs[4,4,5] indicateurs[4,4,6]
 indicateurs[4,4,7] indicateurs[4,4,8] indicateurs[4,4,9]
 indicateurs[4,5,1] indicateurs[4,5,2] indicateurs[4,5,3]
 indicateurs[4,5,4] indicateurs[4,5,5] indicateurs[4,5,6]
 indicateurs[4,5,7] indicateurs[4,5,8] indicateurs[4,5,9]
 indicateurs[4,6,1] indicateurs[4,6,2] indicateurs[4,6,3]
 indicateurs[4,6,4] indicateurs[4,6,5] indicateurs[4,6,6]
 indicateurs[4,6,7] indicateurs[4,6,8] indicateurs[4,6,9]
 indicateurs[4,7,1] indicateurs[4,7,2] indicateurs[4,7,3]
 indicateurs[4,7,4] indicateurs[4,7,5] indicateurs[4,7,6]
 indicateurs[4,7,7] indicateurs[4,7,8] indicateurs[4,7,9]
 indicateurs[4,8,1] indicateurs[4,8,2] indicateurs[4,8,3]
 indicateurs[4,8,4] indicateurs[4,8,5] indicateurs[4,8,6]
 indicateurs[4,8,7] indicateurs[4,8,8] indicateurs[4,8,9]
 indicateurs[4,9,1] indicateurs[4,9,2] indicateurs[4,9,3]
 indicateurs[4,9,4] indicateurs[4,9,5] indicateurs[4,9,6]
 indicateurs[4,9,7] indicateurs[4,9,8] indicateurs[4,9,9]
 indicateurs[5,1,1] indicateurs[5,1,2] indicateurs[5,1,3]
 indicateurs[5,1,4] indicateurs[5,1,5] indicateurs[5,1,6]
 indicateurs[5,1,7] indicateurs[5,1,8] indicateurs[5,1,9]
 indicateurs[5,2,1] indicateurs[5,2,2] indicateurs[5,2,3]
 indicateurs[5,2,4] indicateurs[5,2,5] indicateurs[5,2,6]
 indicateurs[5,2,7] indicateurs[5,2,8] indicateurs[5,2,9]
 indicateurs[5,3,1] indicateurs[5,3,2] indicateurs[5,3,3]
 indicateurs[5,3,4] indicateurs[5,3,5] indicateurs[5,3,6]
 indicateurs[5,3,7] indicateurs[5,3,8] indicateurs[5,3,9]
 indicateurs[5,4,1] indicateurs[5,4,2] indicateurs[5,4,3]
 indicateurs[5,4,4] indicateurs[5,4,5] indicateurs[5,4,6]
 indicateurs[5,4,7] indicateurs[5,4,8] indicateurs[5,4,9]
 indicateurs[5,5,1] indicateurs[5,5,2] indicateurs[5,5,3]
 indicateurs[5,5,4] indicateurs[5,5,5] indicateurs[5,5,6]
 indicateurs[5,5,7] indicateurs[5,5,8] indicateurs[5,5,9]
 indicateurs[5,6,1] indicateurs[5,6,2] indicateurs[5,6,3]
 indicateurs[5,6,4] indicateurs[5,6,5] indicateurs[5,6,6]
 indicateurs[5,6,7] indicateurs[5,6,8] indicateurs[5,6,9]
 indicateurs[5,7,1] indicateurs[5,7,2] indicateurs[5,7,3]
 indicateurs[5,7,4] indicateurs[5,7,5] indicateurs[5,7,6]
 indicateurs[5,7,7] indicateurs[5,7,8] indicateurs[5,7,9]
 indicateurs[5,8,1] indicateurs[5,8,2] indicateurs[5,8,3]
 indicateurs[5,8,4] indicateurs[5,8,5] indicateurs[5,8,6]
 indicateurs[5,8,7] indicateurs[5,8,8] indicateurs[5,8,9]
 indicateurs[5,9,1] indicateurs[5,9,2] indicateurs[5,9,3]
 indicateurs[5,9,4] indicateurs[5,9,5] indicateurs[5,9,6]
 indicateurs[5,9,7] indicateurs[5,9,8] indicateurs[5,9,9]
 indicateurs[6,1,1] indicateurs[6,1,2] indicateurs[6,1,3]
 indicateurs[6,1,4] indicateurs[6,1,5] indicateurs[6,1,6]
 indicateurs[6,1,7] indicateurs[6,1,8] indicateurs[6,1,9]
 indicateurs[6,2,1] indicateurs[6,2,2] indicateurs[6,2,3]
 indicateurs[6,2,4] indicateurs[6,2,5] indicateurs[6,2,6]
 indicateurs[6,2,7] indicateurs[6,2,8] indicateurs[6,2,9]
 indicateurs[6,3,1] indicateurs[6,3,2] indicateurs[6,3,3]
 indicateurs[6,3,4] indicateurs[6,3,5] indicateurs[6,3,6]
 indicateurs[6,3,7] indicateurs[6,3,8] indicateurs[6,3,9]
 indicateurs[6,4,1] indicateurs[6,4,2] indicateurs[6,4,3]
 indicateurs[6,4,4] indicateurs[6,4,5] indicateurs[6,4,6]
 indicateurs[6,4,7] indicateurs[6,4,8] indicateurs[6,4,9]
 indicateurs[6,5,1] indicateurs[6,5,2] indicateurs[6,5,3]
 indicateurs[6,5,4] indicateurs[6,5,5] indicateurs[6,5,6]
 indicateurs[6,5,7] indicateurs[6,5,8] indicateurs[6,5,9]
 indicateurs[6,6,1] indicateurs[6,6,2] indicateurs[6,6,3]
 indicateurs[6,6,4] indicateurs[6,6,5] indicateurs[6,6,6]
 indicateurs[6,6,7] indicateurs[6,6,8] indicateurs[6,6,9]
 indicateurs[6,7,1] indicateurs[6,7,2] indicateurs[6,7,3]
 indicateurs[6,7,4] indicateurs[6,7,5] indicateurs[6,7,6]
 indicateurs[6,7,7] indicateurs[6,7,8] indicateurs[6,7,9]
 indicateurs[6,8,1] indicateurs[6,8,2] indicateurs[6,8,3]
 indicateurs[6,8,4] indicateurs[6,8,5] indicateurs[6,8,6]
 indicateurs[6,8,7] indicateurs[6,8,8] indicateurs[6,8,9]
 indicateurs[6,9,1] indicateurs[6,9,2] indicateurs[6,9,3]
 indicateurs[6,9,4] indicateurs[6,9,5] indicateurs[6,9,6]
 indicateurs[6,9,7] indicateurs[6,9,8] indicateurs[6,9,9]
 indicateurs[7,1,1] indicateurs[7,1,2] indicateurs[7,1,3]
 indicateurs[7,1,4] indicateurs[7,1,5] indicateurs[7,1,6]
 indicateurs[7,1,7] indicateurs[7,1,8] indicateurs[7,1,9]
 indicateurs[7,2,1] indicateurs[7,2,2] indicateurs[7,2,3]
 indicateurs[7,2,4] indicateurs[7,2,5] indicateurs[7,2,6]
 indicateurs[7,2,7] indicateurs[7,2,8] indicateurs[7,2,9]
 indicateurs[7,3,1] indicateurs[7,3,2] indicateurs[7,3,3]
 indicateurs[7,3,4] indicateurs[7,3,5] indicateurs[7,3,6]
 indicateurs[7,3,7] indicateurs[7,3,8] indicateurs[7,3,9]
 indicateurs[7,4,1] indicateurs[7,4,2] indicateurs[7,4,3]
 indicateurs[7,4,4] indicateurs[7,4,5] indicateurs[7,4,6]
 indicateurs[7,4,7] indicateurs[7,4,8] indicateurs[7,4,9]
 indicateurs[7,5,1] indicateurs[7,5,2] indicateurs[7,5,3]
 indicateurs[7,5,4] indicateurs[7,5,5] indicateurs[7,5,6]
 indicateurs[7,5,7] indicateurs[7,5,8] indicateurs[7,5,9]
 indicateurs[7,6,1] indicateurs[7,6,2] indicateurs[7,6,3]
 indicateurs[7,6,4] indicateurs[7,6,5] indicateurs[7,6,6]
 indicateurs[7,6,7] indicateurs[7,6,8] indicateurs[7,6,9]
 indicateurs[7,7,1] indicateurs[7,7,2] indicateurs[7,7,3]
 indicateurs[7,7,4] indicateurs[7,7,5] indicateurs[7,7,6]
 indicateurs[7,7,7] indicateurs[7,7,8] indicateurs[7,7,9]
 indicateurs[7,8,1] indicateurs[7,8,2] indicateurs[7,8,3]
 indicateurs[7,8,4] indicateurs[7,8,5] indicateurs[7,8,6]
 indicateurs[7,8,7] indicateurs[7,8,8] indicateurs[7,8,9]
 indicateurs[7,9,1] indicateurs[7,9,2] indicateurs[7,9,3]
 indicateurs[7,9,4] indicateurs[7,9,5] indicateurs[7,9,6]
 indicateurs[7,9,7] indicateurs[7,9,8] indicateurs[7,9,9]
 indicateurs[8,1,1] indicateurs[8,1,2] indicateurs[8,1,3]
 indicateurs[8,1,4] indicateurs[8,1,5] indicateurs[8,1,6]
 indicateurs[8,1,7] indicateurs[8,1,8] indicateurs[8,1,9]
 indicateurs[8,2,1] indicateurs[8,2,2] indicateurs[8,2,3]
 indicateurs[8,2,4] indicateurs[8,2,5] indicateurs[8,2,6]
 indicateurs[8,2,7] indicateurs[8,2,8] indicateurs[8,2,9]
 indicateurs[8,3,1] indicateurs[8,3,2] indicateurs[8,3,3]
 indicateurs[8,3,4] indicateurs[8,3,5] indicateurs[8,3,6]
 indicateurs[8,3,7] indicateurs[8,3,8] indicateurs[8,3,9]
 indicateurs[8,4,1] indicateurs[8,4,2] indicateurs[8,4,3]
 indicateurs[8,4,4] indicateurs[8,4,5] indicateurs[8,4,6]
 indicateurs[8,4,7] indicateurs[8,4,8] indicateurs[8,4,9]
 indicateurs[8,5,1] indicateurs[8,5,2] indicateurs[8,5,3]
 indicateurs[8,5,4] indicateurs[8,5,5] indicateurs[8,5,6]
 indicateurs[8,5,7] indicateurs[8,5,8] indicateurs[8,5,9]
 indicateurs[8,6,1] indicateurs[8,6,2] indicateurs[8,6,3]
 indicateurs[8,6,4] indicateurs[8,6,5] indicateurs[8,6,6]
 indicateurs[8,6,7] indicateurs[8,6,8] indicateurs[8,6,9]
 indicateurs[8,7,1] indicateurs[8,7,2] indicateurs[8,7,3]
 indicateurs[8,7,4] indicateurs[8,7,5] indicateurs[8,7,6]
 indicateurs[8,7,7] indicateurs[8,7,8] indicateurs[8,7,9]
 indicateurs[8,8,1] indicateurs[8,8,2] indicateurs[8,8,3]
 indicateurs[8,8,4] indicateurs[8,8,5] indicateurs[8,8,6]
 indicateurs[8,8,7] indicateurs[8,8,8] indicateurs[8,8,9]
 indicateurs[8,9,1] indicateurs[8,9,2] indicateurs[8,9,3]
 indicateurs[8,9,4] indicateurs[8,9,5] indicateurs[8,9,6]
 indicateurs[8,9,7] indicateurs[8,9,8] indicateurs[8,9,9]
 indicateurs[9,1,1] indicateurs[9,1,2] indicateurs[9,1,3]
 indicateurs[9,1,4] indicateurs[9,1,5] indicateurs[9,1,6]
 indicateurs[9,1,7] indicateurs[9,1,8] indicateurs[9,1,9]
 indicateurs[9,2,1] indicateurs[9,2,2] indicateurs[9,2,3]
 indicateurs[9,2,4] indicateurs[9,2,5] indicateurs[9,2,6]
 indicateurs[9,2,7] indicateurs[9,2,8] indicateurs[9,2,9]
 indicateurs[9,3,1] indicateurs[9,3,2] indicateurs[9,3,3]
 indicateurs[9,3,4] indicateurs[9,3,5] indicateurs[9,3,6]
 indicateurs[9,3,7] indicateurs[9,3,8] indicateurs[9,3,9]
 indicateurs[9,4,1] indicateurs[9,4,2] indicateurs[9,4,3]
 indicateurs[9,4,4] indicateurs[9,4,5] indicateurs[9,4,6]
 indicateurs[9,4,7] indicateurs[9,4,8] indicateurs[9,4,9]
 indicateurs[9,5,1] indicateurs[9,5,2] indicateurs[9,5,3]
 indicateurs[9,5,4] indicateurs[9,5,5] indicateurs[9,5,6]
 indicateurs[9,5,7] indicateurs[9,5,8] indicateurs[9,5,9]
 indicateurs[9,6,1] indicateurs[9,6,2] indicateurs[9,6,3]
 indicateurs[9,6,4] indicateurs[9,6,5] indicateurs[9,6,6]
 indicateurs[9,6,7] indicateurs[9,6,8] indicateurs[9,6,9]
 indicateurs[9,7,1] indicateurs[9,7,2] indicateurs[9,7,3]
 indicateurs[9,7,4] indicateurs[9,7,5] indicateurs[9,7,6]
 indicateurs[9,7,7] indicateurs[9,7,8] indicateurs[9,7,9]
 indicateurs[9,8,1] indicateurs[9,8,2] indicateurs[9,8,3]
 indicateurs[9,8,4] indicateurs[9,8,5] indicateurs[9,8,6]
 indicateurs[9,8,7] indicateurs[9,8,8] indicateurs[9,8,9]
 indicateurs[9,9,1] indicateurs[9,9,2] indicateurs[9,9,3]
 indicateurs[9,9,4] indicateurs[9,9,5] indicateurs[9,9,6]
 indicateurs[9,9,7] indicateurs[9,9,8] indicateurs[9,9,9]
End
