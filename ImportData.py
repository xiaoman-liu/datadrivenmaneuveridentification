import os
import pickle
from pathlib import Path
import math
from natsort import natsorted
#import pandas as pd

# import sys
# sys.path.append(r'/home/xinjie/xiaoman/codes/datadrivenmaneuveridentification-Xiaoman/scenarioSubclasses.py')
#
# with open("/home/xinjie/Desktop/newformat/1448.p","rb") as f:
#     b=pickle.load(f)

path = Path(os.getcwd()) / "data"

pickle_in = open(path / "scenarios", "rb")
data = pickle.load(pickle_in)
# a = open("/home/xinjie/xiaoman/codes/datadrivenmaneuveridentification-Xiaoman/data/newformat/1200.p","rb")
# data = pd.read_pickle("/home/xinjie/xiaoman/codes/datadrivenmaneuveridentification-Xiaoman/data/newformat/1200.p")
# TODO: sort firstly



"""
case 1:
specify keys for data_scenario if you just want to run in single scenario

case 2:
set key to None, if you want to run top_k_scenario
set top_k_scenario to k to select top k scenarios from data
set top_k_scenario to -1 to select all scenarios from data
"""

scenarios_data = []
sample_num = 0
top_k_scenario = 186 # total scenarios = 244
# keys = None
scenario_num = 0
scenario_list = []#[201,202,203,205,206,207,208,209,210,211,212,213,214,215,216,218,221,222,223,224,225,226,228,229,230,231,232,233,234,235,236,237,238,239]
scenario_keys = "%s"%scenario_list
scenarios_name = []
# keys = ["scenario" + "%s"%i for i in scenario_list if scenario_list]
keys = ['scenario197', 'scenario199', 'scenario212',
'scenario214', 'scenario215', 'scenario223', 'scenario224', 'scenario225', 'scenario226',
'scenario231', 'scenario238', 'scenario240', 'scenario252', 'scenario256', 'scenario257',
'scenario259', 'scenario261', 'scenario267']
##### for sample type statistic
# scenario_ped = set()
# from collections import Counter
# sample_type = []
if scenario_list:
    for key in keys:
        scenario_data = data[key]
        scenarios_data.append(scenario_data)
        print("loading data from {}".format(key))
        scenario_num += 1
        for key in scenario_data:
            sample_num += 1

else:
    for i, scenario in enumerate(natsorted(data.keys())):
        if i == top_k_scenario: break
        print("loading data from {}".format(scenario))
        scenario_num +=1


        scenario_data = data[scenario]
        ##### for sample type statistic
        # for sample_name in scenario_data:
        #     sample_type.append(scenario_data[sample_name]["Type"])
        #     if scenario_data[sample_name]["Type"] == "mbike":
        #         scenario_ped.add(scenario)
        scenarios_data.append(scenario_data)
        for key in scenario_data:
            sample_num += 1
            scenarios_name.append(scenario)



##### for sample type statistic
# print(Counter(sample_type))
# print(scenario_ped)

print("total sample number: {}".format(sample_num))
print("total scenerios number: {}".format(scenario_num))


"""
(['scenario213', 'scenario85', 'scenario2', 'scenario15', 'scenario23', 'scenario235', 
'scenario267', 'scenario121', 'scenario231', 'scenario257', 'scenario39', 'scenario87', 
'scenario115', 'scenario177', 'scenario60', 'scenario173', 'scenario3', 'scenario166', 
'scenario246', 'scenario225', 'scenario153', 'scenario210', 'scenario127', 'scenario164', 
'scenario102', 'scenario143', 'scenario76', 'scenario160', 'scenario202', 'scenario226', 
'scenario259', 'scenario242', 'scenario243', 'scenario244', 'scenario197', 'scenario47', 
'scenario103', 'scenario187', 'scenario93', 'scenario108', 'scenario57', 'scenario58', 
'scenario64', 'scenario80', 'scenario112', 'scenario247', 'scenario240', 'scenario174', 
'scenario263', 'scenario1', 'scenario71', 'scenario72', 'scenario59', 'scenario62', 
'scenario201', 'scenario5', 'scenario99', 'scenario36', 'scenario73', 'scenario16', 
'scenario122', 'scenario142', 'scenario61', 'scenario66', 'scenario35', 'scenario46', 
'scenario107', 'scenario20', 'scenario48', 'scenario49', 'scenario34', 'scenario50', 
'scenario123', 'scenario63', 'scenario65', 'scenario69', 'scenario4', 'scenario55', 
'scenario82', 'scenario83', 'scenario124', 'scenario84', 'scenario116', 'scenario86', 
'scenario113', 'scenario106', 'scenario105', 'scenario91', 'scenario92', 'scenario94', 
'scenario88', 'scenario101', 'scenario100', 'scenario17', 'scenario109', 'scenario110', 
'scenario111', 'scenario114', 'scenario89', 'scenario162', 'scenario75', 'scenario145', 
'scenario232', 'scenario52', 'scenario78', 'scenario54', 'scenario222', 'scenario158', 
'scenario167', 'scenario96', 'scenario265', 'scenario179', 'scenario56', 'scenario183', 
'scenario38', 'scenario159', 'scenario181', 'scenario182', 'scenario70', 'scenario128', 
'scenario129', 'scenario180', 'scenario131', 'scenario97', 'scenario119', 'scenario120', 
'scenario79', 'scenario74', 'scenario81', 'scenario19', 'scenario10', 'scenario21', 
'scenario140', 'scenario22', 'scenario24', 'scenario25', 'scenario28', 'scenario29', 
'scenario31', 'scenario172', 'scenario32', 'scenario195', 'scenario132', 'scenario133', 
'scenario236', 'scenario134', 'scenario141', 'scenario138', 'scenario136', 'scenario137', 
'scenario139', 'scenario146', 'scenario40', 'scenario41', 'scenario42', 'scenario43', 
'scenario44', 'scenario45', 'scenario51', 'scenario53', 'scenario125', 'scenario13', 
'scenario149', 'scenario192', 'scenario90', 'scenario193', 'scenario170', 'scenario150', 
'scenario165', 'scenario144', 'scenario151', 'scenario154', 'scenario184', 'scenario191', 
'scenario189', 'scenario147', 'scenario168', 'scenario155', 'scenario148', 'scenario135', 
'scenario169', 'scenario7', 'scenario157', 'scenario175', 'scenario130', 'scenario249', 
'scenario14', 'scenario176', 'scenario205', 'scenario178', 'scenario196', 'scenario198', 
'scenario199', 'scenario203', 'scenario118', 'scenario12', 'scenario211', 'scenario212', 
'scenario209', 'scenario208', 'scenario186', 'scenario214', 'scenario215', 'scenario216', 
'scenario126', 'scenario221', 'scenario223', 'scenario224', 'scenario266', 'scenario230', 
'scenario194', 'scenario228', 'scenario156', 'scenario152', 'scenario77', 'scenario207', 
'scenario163', 'scenario161', 'scenario6', 'scenario8', 'scenario9', 'scenario11', 
'scenario250', 'scenario229', 'scenario188', 'scenario233', 'scenario234', 'scenario237', 
'scenario238', 'scenario239', 'scenario248', 'scenario30', 'scenario251', 'scenario252', 
'scenario256', 'scenario258', 'scenario260', 'scenario261', 'scenario262', 'scenario255', 
'scenario206', 'scenario98', 'scenario264', 'scenario218'])

sample_type_set = {'car': 526, 'ped': 90, 'bicy': 34, 'truck': 32, 'mbike': 17}

scenario contains ped

['scenario124', 'scenario127', 'scenario240', 'scenario182', 'scenario129', 'scenario155', 
'scenario29', 'scenario46', 'scenario261', 'scenario231', 'scenario259', 'scenario156', 
'scenario120', 'scenario214', 'scenario238', 'scenario257', 'scenario45', 'scenario109', 
'scenario252', 'scenario5', 'scenario122', 'scenario4', 'scenario101', 'scenario223', 
'scenario224', 'scenario30', 'scenario181', 'scenario42', 'scenario225', 'scenario43', 
'scenario165', 'scenario197', 'scenario212', 'scenario50', 'scenario55', 'scenario79', 
'scenario77', 'scenario57', 'scenario53', 'scenario28', 'scenario87', 'scenario86', 
'scenario215', 'scenario100', 'scenario56', 'scenario199', 'scenario92', 'scenario256', 
'scenario6', 'scenario226', 'scenario267', 'scenario38', 'scenario24', 'scenario23', 
'scenario41', 'scenario39', 'scenario191', 'scenario188', 'scenario93', 'scenario47', 
'scenario7', 'scenario121', 'scenario78']

scenario contains bicy
{'scenario44', 'scenario211', 'scenario61', 'scenario256', 'scenario191', 'scenario45', 
'scenario201', 'scenario198', 'scenario62', 'scenario258', 'scenario94', 'scenario23', 
'scenario83', 'scenario252', 'scenario53', 'scenario55', 'scenario267', 'scenario36', 
'scenario189', 'scenario54', 'scenario63', 'scenario60', 'scenario42', 'scenario235', 
'scenario209', 'scenario210', 'scenario43', 'scenario24'}

scenario contains truck
{'scenario119', 'scenario56', 'scenario149', 'scenario240', 'scenario148', 'scenario260', 
'scenario29', 'scenario238', 'scenario28', 'scenario80', 'scenario242', 'scenario213', 
'scenario30', 'scenario256', 'scenario267', 'scenario105', 'scenario66', 'scenario255', 
'scenario55', 'scenario53', 'scenario188', 'scenario209', 'scenario65', 'scenario134'}

scenario contains mbike
{'scenario39', 'scenario195', 'scenario50', 'scenario51', 'scenario66', 'scenario65', 
'scenario17', 'scenario96', 'scenario52', 'scenario31', 'scenario215', 'scenario29', 
'scenario214', 'scenario38', 'scenario93', 'scenario30', 'scenario201'}



after natsort
scenario contains ped
['scenario4', 'scenario5', 'scenario6', 'scenario7', 'scenario23', 'scenario24', 
'scenario28', 'scenario29', 'scenario30', 'scenario38', 'scenario39', 'scenario41', 
'scenario42', 'scenario43', 'scenario45', 'scenario46', 'scenario47', 'scenario50', 
'scenario53', 'scenario55', 'scenario56', 'scenario57', 'scenario77', 'scenario78', 
'scenario79', 'scenario86', 'scenario87', 'scenario92', 'scenario93', 'scenario100', 
'scenario101', 'scenario109', 'scenario120', 'scenario121', 'scenario122', 'scenario124', 
'scenario127', 'scenario129', 'scenario155', 'scenario156', 'scenario165', 'scenario181', 
'scenario182', 'scenario188', 'scenario191', 'scenario197', 'scenario199', 'scenario212', 
'scenario214', 'scenario215', 'scenario223', 'scenario224', 'scenario225', 'scenario226', 
'scenario231', 'scenario238', 'scenario240', 'scenario252', 'scenario256', 'scenario257', 
'scenario259', 'scenario261', 'scenario267']


scenario contains bicy
['scenario23', 'scenario24', 'scenario36', 'scenario42', 'scenario43', 'scenario44', 
'scenario45', 'scenario53', 'scenario54', 'scenario55', 'scenario60', 'scenario61', 
'scenario62', 'scenario63', 'scenario83', 'scenario94', 'scenario189', 'scenario191', 
'scenario198', 'scenario201', 'scenario209', 'scenario210', 'scenario211', 'scenario235', 
'scenario252', 'scenario256', 'scenario258', 'scenario267']


scenario contains truck
['scenario28', 'scenario29', 'scenario30', 'scenario53', 'scenario55', 'scenario56', 
'scenario65', 'scenario66', 'scenario80', 'scenario105', 'scenario119', 'scenario134', 
'scenario148', 'scenario149', 'scenario188', 'scenario209', 'scenario213', 'scenario238', 
'scenario240', 'scenario242', 'scenario255', 'scenario256', 'scenario260', 'scenario267']


scenario contains mbike
['scenario17', 'scenario29', 'scenario30', 'scenario31', 'scenario38', 'scenario39', 
'scenario50', 'scenario51', 'scenario52', 'scenario65', 'scenario66', 'scenario93', 
'scenario96', 'scenario195', 'scenario201', 'scenario214', 'scenario215']

"""