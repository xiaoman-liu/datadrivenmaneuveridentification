longitudinalBasicManeuverIdMapping = {
    -1: "NoneLong",
    1: "CruiseFree",
    2: "Follow",
    3: "Approach",
    4: "Stop",
    5: "StandStill"
}

lateralBasicManeuverIdMapping = {
    -1: "NoneLat",
    1: "HoldLane",
    2: "ChangeLane",
    3: "CrossJunction",
    31: "CrossStraight",
    32: "TurnLeft",
    33: "TurnRight",
    4: "CrossRoad"
}

lateral_distribution = {
    0: "NoneLat",
    1: "HoldLane",
    2: "ChangeLane",
    3: "CrossJunction",
    4: "CrossStraight",
    5: "TurnLeft",
    6: "TurnRight",
    7: "CrossRoad"
}

roadID = [   0, 1000000, 1001000, 1002000, 1003000, 1004000,
       1005000, 1006000, 1007000, 1008000, 1009000, 1010000,
       1011000, 1012000, 1013000, 1014000, 1015000, 1016000,
       1017000, 1018000, 1019000, 1020000, 1024000, 1025000,
       1028000, 1030000, 1032000, 1033000, 3000020, 3000021,
       3000023, 3000025, 3001020, 3001021, 3001022, 3001023,
       3001024, 3002020, 3002021, 3002022, 3002023, 3003020,
       3003021, 3005020, 3005021, 3005022, 3007020, 3007021,
       3007022, 3007400, 3009020, 3009021, 3009022, 3011020,
       3011021, 3011023, 3011025, 3012021, 3012022, 3012024,
       3012025, 3012028, 3012030, 3012031, 3013020, 3013021,
       3014020, 3014021, 3014022, 3014024, 3014025, 3015020,
       3015021, 3016020, 3016021, 3017020, 3017021, 3017022,
       3017023, 3017025, 3018020, 3018021, 3019020, 3019021,
       3020020, 3020021, 3020400, 3024020] # 88

laneID = [-6., -5., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,
        8.,  9., 99.] # 16



sample_type_set = {'car': 0, 'ped': 1, 'bicy': 2, 'truck': 3, 'mbike': 4}



longitudinal_id_mapping = {key: index for index, key in enumerate(longitudinalBasicManeuverIdMapping)}
lateral_id_mapping = {key: index for index, key in enumerate(lateralBasicManeuverIdMapping)}
# print(lateral_id_mapping)

roadID_mapping = {
    key:value for value,key in enumerate(roadID)
}

laneID_mapping = {
    key:value for value, key in enumerate(laneID)
}

len_ROADID = len(roadID)
len_LANEID = len(laneID)
# section_roadid = {"":1,"":2,"3":3,"":4,"":5,"":6,"":7,"":8,"":9,"":10,
# #                   "":11,"":12,"":13,"":14,"":15,"":16,"":17,"":18,"":19,"":20,
# #                   "":21,"":22,"":23,"":24,"":25,"":26,"":27}
section_roadid = {}
for key,value in enumerate(roadID):
    if key in range(1,28):
        value = str(value)
        section_roadid[value] = key


print