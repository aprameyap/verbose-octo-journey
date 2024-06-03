# verbose-octo-journey

Forecasting prices of Henry Hub Natural Gas Futures (NYMEX:NG1!)

There is an optimal number of weeks (N) to which the model predicts the cumulative percentage change to the given trading logic.
The hypothesis would be (check N_finder.py for N calculation): The N value remains constant in near future, the new N value will be searched by running the same backtest through the future. N would also change with the max amount of the aum one is willing to risk, since the simulator doesn't allow trades to take place once open positions exhaust the total aum.\
N_finder.py is a a hit and trial method for finding the best N for your backtesting period.
In the current test set, N ~ (36-37) and Risk > 0.5 gives the best results. All of this has a threshold of 15% (up or down) for each trade to take place, this can be treated as a variable and optimised also, but finding an optimum threshold will create unnecessary computational complexities.
Adding more features and increasing the time horizon will increase the model's accuracy. The current model is definitely not suited for real-world applications, but just a demonstration.
The trading logic and simulator, though, are over-simplified and have very little correlation on how a real system would work.

Update (03-06-2024): I recreated the dataset (in NG_new), the old, small dataset still exists in NG directory (for the sake of showing progress :)). The new dataset now goes back .till 1994, compared 2010 earlier. Lag features were also added to NG1 price, Crude oil price, imports, exports and CPI

!['Figure_1.png']
!['Figure_2.png']
!['Figure_3.png']


Training info:

03-06-2024 (Changed up all of the features, added 4 weeks lag, removed features which didn't go to the 90s, thus increasing the size of the dataset by 3x)
```
Mean Absolute Error: 0.2214295849120274
Root Mean Squared Error: 0.3452974706866467
                                         Feature  Importance
20  Natural Gas Futures Contract 1 $/MMBTU_Lag_1   21.788647
21  Natural Gas Futures Contract 1 $/MMBTU_Lag_2    8.502423
22  Natural Gas Futures Contract 1 $/MMBTU_Lag_3    5.638664
23  Natural Gas Futures Contract 1 $/MMBTU_Lag_4    4.758264
27                                          wind    3.739596
8                          Crude_Oil_Price_Lag_3    3.727448
5                                Crude_Oil_Price    3.466177
16                            Imports_MMcf_Lag_1    3.054084
24                      hurricane_force_diameter    3.040696
7                          Crude_Oil_Price_Lag_2    3.031803
2                                      CPI_Lag_2    2.994720
13                            Exports_MMcf_Lag_3    2.964022
9                          Crude_Oil_Price_Lag_4    2.957043
1                                      CPI_Lag_1    2.907031
0                                            CPI    2.854183
6                          Crude_Oil_Price_Lag_1    2.724830
4                                      CPI_Lag_4    2.581632
3                                      CPI_Lag_3    2.397774
26                  tropicalstorm_force_diameter    2.388883
17                            Imports_MMcf_Lag_2    2.143555
25                                      pressure    2.025327
19                            Imports_MMcf_Lag_4    2.006282
14                            Exports_MMcf_Lag_4    1.845286
11                            Exports_MMcf_Lag_1    1.692056
15                                  Imports_MMcf    1.628155
18                            Imports_MMcf_Lag_3    1.249912
10                                  Exports_MMcf    1.057670
12                            Exports_MMcf_Lag_2    0.833839
```


02-06-2024 (Added exports as a feature)
```
Mean Absolute Error: 0.21148400411530524
Root Mean Squared Error: 0.2946208583556225
                                Feature  Importance
1   Cushing OK WTI Spot Price FOB $/bbl   21.756369
15                                 Salt    8.932936
2                              DTWEXBGS    8.428988
5     Europe Brent Spot Price FOB $/bbl    8.355335
6                          Exports_MMcf    7.180510
0                  CORESTICKM159SFRBATL    6.667941
8                          Imports_MMcf    6.098091
16                 South Central Region    4.886378
11                      Mountain Region    4.828341
13                       Pacific Region    3.841371
10                       Midwest Region    3.032261
3       Dry_Natural_Gas_Production_MMcf    2.891638
4                           East Region    2.849203
14        Plant_Liquids_Production_MMcf    2.830410
12                              NonSalt    2.309049
7                Gross_Withdrawals_MMcf    1.458910
9              Marketed_Production_MMcf    1.452009
18                             pressure    1.051803
19         tropicalstorm_force_diameter    0.527162
20                                 wind    0.433531
17             hurricane_force_diameter    0.187765
```

29-05-24 (This is on weekly data, too less data for monthly training)
```
Training:

Mean Absolute Error: 0.2933105616772876
Root Mean Squared Error: 0.8982398909067726
                                Feature  Importance
1   Cushing OK WTI Spot Price FOB $/bbl   15.687328
0                  CORESTICKM159SFRBATL   12.888051
15                 South Central Region    8.243944
10                      Mountain Region    7.161725
7                          Imports_MMcf    7.042360
5     Europe Brent Spot Price FOB $/bbl    6.899478
13        Plant_Liquids_Production_MMcf    6.254457
2                              DTWEXBGS    5.626007
9                        Midwest Region    5.414369
14                                 Salt    3.777512
12                       Pacific Region    3.752354
3       Dry_Natural_Gas_Production_MMcf    3.477094
4                           East Region    3.159954
6                Gross_Withdrawals_MMcf    2.367753
8              Marketed_Production_MMcf    1.963989
11                              NonSalt    1.837040
19                                 wind    1.646050
18         tropicalstorm_force_diameter    1.168211
17                             pressure    1.009984
16             hurricane_force_diameter    0.622340
```

27-05-2024
```

Training:

Mean Absolute Error: 0.4375622595007447
Root Mean Squared Error: 0.5962166281677603
                                Feature  Importance
1   Cushing OK WTI Spot Price FOB $/bbl   24.258784
5     Europe Brent Spot Price FOB $/bbl   18.464282
0                  CORESTICKM159SFRBATL    9.588419
2                              DTWEXBGS    8.046189
11        Plant_Liquids_Production_MMcf    6.747634
6                Gross_Withdrawals_MMcf    6.438796
12                                 Salt    4.421382
7                          Imports_MMcf    4.404972
3       Dry_Natural_Gas_Production_MMcf    3.339133
10                       Pacific Region    2.986192
4                           East Region    2.870208
8              Marketed_Production_MMcf    2.764386
9                       Mountain Region    2.167325
15         tropicalstorm_force_diameter    1.123334
14                             pressure    1.018904
16                                 wind    0.870621
13             hurricane_force_diameter    0.489442
```

23-05-24
```

Training:

Mean Absolute Error: 0.3565561787495663
Root Mean Squared Error: 0.45859897346198075
                                Feature  Importance
1   Cushing OK WTI Spot Price FOB $/bbl   22.612712
5     Europe Brent Spot Price FOB $/bbl   11.676842
12                                 Salt   10.036134
11        Plant_Liquids_Production_MMcf    6.993827
0                  CORESTICKM159SFRBATL    6.885862
9                       Mountain Region    6.545815
4                           East Region    6.443635
7                          Imports_MMcf    6.077576
10                       Pacific Region    5.804332
8              Marketed_Production_MMcf    5.155749
2                              DTWEXBGS    4.892531
6                Gross_Withdrawals_MMcf    4.517890
3       Dry_Natural_Gas_Production_MMcf    2.357097
```

22-05-24: 
```
Training:

Mean Absolute Error: 0.3829875579601106
Root Mean Squared Error: 0.48223190373640573
                               Feature  Importance
1  Cushing OK WTI Spot Price FOB $/bbl   18.375541
4    Europe Brent Spot Price FOB $/bbl   16.635741
6                         Imports_MMcf   15.156283
8        Plant_Liquids_Production_MMcf   12.274401
0                 CORESTICKM159SFRBATL   12.266013
2                             DTWEXBGS    8.789367
7             Marketed_Production_MMcf    7.753190
3      Dry_Natural_Gas_Production_MMcf    5.150494
5               Gross_Withdrawals_MMcf    3.598970
```

Data sources:
- https://fred.stlouisfed.org/series/MEDCPIM158SFRBCLE
- https://www.eia.gov/dnav/ng/ng_move_impc_s1_m.htm
- https://ir.eia.gov/ngs/ngs.html
- https://www.eia.gov/dnav/ng/ng_pri_fut_s1_w.htm
- https://www.eia.gov/dnav/ng/ng_prod_sum_dc_NUS_MMCF_m.htm
- https://www.eia.gov/dnav/ng/hist/rngwhhdm.htm
- https://www.eia.gov/dnav/ng/ng_move_expc_s1_m.htm
- https://fred.stlouisfed.org/series/DTWEXBGS
- https://www.kaggle.com/datasets/utkarshx27/noaa-atlantic-hurricane-database
