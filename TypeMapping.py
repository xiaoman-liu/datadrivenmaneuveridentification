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
longtudinal_distribution = {
    0: "NoneLong",
    1: "CruiseFree",
    2: "Follow",
    3: "Approach",
    4: "Stop",
    5: "StandStill"
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
print(longitudinal_id_mapping)
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


begin_with_changline = [
['car_1027_scenario7', 'car_1059_scenario9', 'car_298_scenario11', 'car_1106_scenario11',
 'car_1106_scenario12', 'car_1659_scenario22', 'bicy_1536_scenario24', 'car_789_scenario24',
 'car_1417_scenario24', 'car_772_scenario24', 'ped_1683_scenario24', 'car_114_scenario28',
 'truck_860_scenario29', 'car_955_scenario34', 'car_971_scenario35', 'bicy_290_scenario36',
 'ped_58_scenario41', 'ped_58_scenario42', 'car_97_scenario43', 'bicy_379_scenario44',
 'car_134_scenario45', 'car_1659_scenario53', 'car_9_scenario53', 'bicy_39_scenario53',
 'car_41_scenario54', 'bicy_811_scenario54', 'car_908_scenario59', 'bicy_232_scenario63',
 'ego_scenario71', 'car_637_scenario74', 'car_200_scenario77', 'car_315_scenario83',
 'ped_810_scenario86', 'ped_810_scenario87', 'car_455_scenario91', 'ego_scenario94',
 'mbike_45_scenario96', 'truck_958_scenario105', 'car_411_scenario113', 'ego_scenario118',
 'ego_scenario132', 'truck_1705_scenario134', 'truck_1703_scenario134', 'car_90_scenario139',
 'truck_2094_scenario148', 'car_347_scenario151', 'car_625_scenario164', 'car_640_scenario165',
 'car_2021_scenario172', 'car_944_scenario178', 'car_1000_scenario181', 'ped_1003_scenario182',
 'car_20_scenario188', 'car_1405_scenario188', 'car_23_scenario188', 'car_31_scenario188',
 'truck_1428_scenario188', 'ego_scenario192', 'car_145_scenario192', 'car_196_scenario194',
 'car_1562_scenario198', 'ped_339_scenario199', 'ped_1611_scenario199'],
['car_456_scenario203', 'car_462_scenario203', 'car_1683_scenario203', 'car_515_scenario206',
 'truck_1850_scenario213', 'ped_2064_scenario224', 'ego_scenario225', 'car_1011_scenario226',
 'car_1002_scenario226', 'car_2262_scenario231', 'car_2323_scenario234', 'car_2602_scenario237',
 'car_2379_scenario237', 'ped_2403_scenario238', 'ego_scenario239', 'truck_667_scenario242',
 'car_23_scenario242', 'car_31_scenario243', 'car_179_scenario249', 'bicy_28_scenario256',
 'truck_20_scenario256', 'car_7_scenario262']

]
bad_cases = {"acc_<0.3":['car_114_scenario28_0.093', 'ped_1003_scenario181_0.107', 'car_31_scenario188_0.166',
                         'ped_121_scenario191_0.000', 'car_196_scenario194_0.055', 'car_1597_scenario199_0.197',
                         'car_515_scenario206_0.118', 'ped_2082_scenario224_0.052', 'car_1002_scenario226_0.006',
                         'car_1267_scenario237_0.144', 'truck_2396_scenario238_0.060', 'car_778_scenario248_0.228',
                         'car_782_scenario248_0.296', 'car_177_scenario249_0.155', 'car_179_scenario249_0.009',
                         'car_216_scenario251_0.251', 'bicy_79_scenario258_0.020', 'car_477_scenario263_0.117'],
            "acc_0.3-0.6":['bicy_1536_scenario24_0.493', 'car_625_scenario164_0.594', 'car_1405_scenario188_0.507',
                            'car_41_scenario189_0.326', 'car_119_scenario191_0.503', 'ego_scenario192_0.473',
                            'ped_1611_scenario199_0.421', 'car_385_scenario201_0.537', 'car_397_scenario202_0.537',
                            'car_456_scenario203_0.536', 'car_462_scenario203_0.471', 'car_1683_scenario203_0.411',
                            'car_504_scenario206_0.574', 'truck_1850_scenario213_0.558', 'car_683_scenario213_0.364',
                            'car_2032_scenario223_0.437', 'car_1110_scenario230_0.401', 'car_1236_scenario235_0.545',
                            'car_2379_scenario237_0.368', 'car_1269_scenario238_0.426', 'car_40_scenario243_0.507',
                            'car_83_scenario244_0.383', 'car_812_scenario249_0.337', 'ped_246_scenario252_0.570',
                            'truck_336_scenario256_0.493', 'bicy_28_scenario256_0.438'],
            "acc_0.6-0.9":['car_119_scenario6_0.719', 'car_1059_scenario9_0.706', 'car_483_scenario17_0.824',
                           'car_497_scenario17_0.842', 'bicy_1536_scenario23_0.789', 'car_789_scenario24_0.700',
                           'car_772_scenario24_0.765', 'ped_1683_scenario24_0.756', 'car_131_scenario29_0.895',
                           'mbike_155_scenario31_0.879', 'car_971_scenario35_0.840', 'bicy_290_scenario36_0.860',
                           'ego_scenario38_0.810', 'ped_516_scenario42_0.854', 'car_289_scenario52_0.896',
                           'car_41_scenario54_0.823', 'bicy_811_scenario54_0.837', 'bicy_72_scenario55_0.887',
                           'truck_816_scenario55_0.825', 'car_293_scenario65_0.892', 'ego_scenario69_0.889',
                           'ego_scenario71_0.860', 'car_77_scenario71_0.893', 'car_200_scenario77_0.846',
                           'car_315_scenario83_0.890', 'ped_810_scenario86_0.863', 'car_455_scenario91_0.803',
                           'ego_scenario118_0.860', 'car_482_scenario118_0.650', 'ego_scenario132_0.860',
                           'car_1428_scenario132_0.737', 'car_2056_scenario137_0.840', 'car_157_scenario143_0.861',
                           'ego_scenario154_0.656', 'car_590_scenario163_0.719', 'car_640_scenario165_0.837',
                           'ego_scenario167_0.695', 'ped_1003_scenario182_0.874', 'car_1797_scenario183_0.794',
                           'car_1797_scenario184_0.658', 'car_1861_scenario186_0.745', 'car_20_scenario188_0.696',
                           'car_23_scenario188_0.873', 'truck_1428_scenario188_0.813', 'car_41_scenario188_0.798',
                           'bicy_68_scenario189_0.650', 'car_50_scenario189_0.761', 'car_64_scenario189_0.812',
                           'car_2641_scenario191_0.690', 'car_136_scenario191_0.721', 'car_145_scenario192_0.834',
                           'ped_339_scenario199_0.770', 'bicy_360_scenario201_0.658', 'car_426_scenario203_0.846',
                           'bicy_590_scenario210_0.730', 'ego_scenario213_0.856', 'car_1837_scenario213_0.757',
                           'car_891_scenario221_0.870', 'ego_scenario224_0.868', 'ped_2064_scenario224_0.863',
                           'ego_scenario225_0.896', 'ped_2118_scenario225_0.830', 'car_1011_scenario226_0.691',
                           'car_2593_scenario231_0.660', 'car_2323_scenario234_0.696', 'car_2602_scenario237_0.838',
                           'car_2353_scenario237_0.603', 'ped_2403_scenario238_0.899', 'ego_scenario239_0.892',
                           'truck_667_scenario242_0.881', 'car_23_scenario242_0.688', 'car_40_scenario242_0.735',
                           'car_31_scenario243_0.811', 'car_764_scenario246_0.821', 'car_134_scenario246_0.865',
                           'car_169_scenario248_0.601', 'car_191_scenario249_0.843', 'car_219_scenario251_0.632',
                           'ego_scenario256_0.771', 'ego_scenario257_0.839', 'car_112_scenario259_0.645',
                           'car_415_scenario259_0.631', 'truck_615_scenario260_0.784', 'car_7_scenario261_0.826',
                           'car_7_scenario262_0.859', 'car_262_scenario267_0.745']

            }