\ Model laminage
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 0.072524059911744 esort_4 - 0.9351111715799409 esort_6
   + 0.2405479777009347 esort_1 + 0.2542764497215684 esort_2
   - 0.3100805673894991 esort_5 - 0.133941382263949 esort_3
   + 40.55541079921366 Constant
Subject To
 esort1_ct: esort_1 <= 25.5
 esort2_ct: - 0.75 esort_1 + esort_2 <= 0
 esort3_ct: - 0.75 esort_2 + esort_3 <= 0
 esort4_ct: esort_4 - 0.75 esort_3 <= 0
 esort5_ct: - 0.75 esort_4 + esort_5 <= 0
 esort6_ct: esort_6 - 0.75 esort_5 <= 0
 esort6_ct2: esort_6 >= 2.50000001
 pmax_f1: - 0.295277905864285 esort_1 <= -1.409753378329324
 pmax_f2: 0.5358258835652198 esort_1 - 0.8574295492821795 esort_2
   <= 5.447931721393919
 pmax_f3: 1.111705999003748 esort_2 - 1.714048652798057 esort_3
   <= 5.274982729606171
 pmax_f4: - 2.423124981470143 esort_4 + 1.580107270534108 esort_3
   <= 3.610368815896264
 pmax_f5: 2.350600921558399 esort_4 - 3.287116689912771 esort_5
   <= 6.095392940601074
 pmax_f6: - 4.124942517226981 esort_6 + 2.977036122523272 esort_5
   <= 3.68159536591245
 pmax_f7: 3.18983134564704 esort_6 <= 11.74407100570578
Bounds
 Constant = 1
End
