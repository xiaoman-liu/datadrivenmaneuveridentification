lane_maneuver = {
    0: "no_lane",
    1: "follow_lane",
    2: "lane_change"
}

lane_id_mapping = {
    key: index for index, key in enumerate(lane_maneuver)
}

"{-1: 0, 1: 1, 2: 2, -2: 3}"

lanemap = {
    0:"no_lane",
    1:"follow_lane",
    2:"lane_change"
    # 3:"filling"
}

vehicle_state_maneuver = {
    1:"accelerate",
    11: "driveaway",
    2: "keep_velocity",
    21: "standstill",
    3: "decelerate",
    31:"halt"
    # 4: "reversing"

}

vehicle_state_id_mapping = {
key: index for index, key in enumerate(vehicle_state_maneuver)
}

vehiclestate_map = {
    0: "accelerate",
    1: "driveaway",
    2: "keep_velocity",
    3: "standstill",
    4: "decelerate",
    5: "halt"
    #6: "reversing"
}

turn_maneuver = {

    # -1: "no_lane",
    0: "no_junction",
    1: "turn_left",
    2: "turn_right"

}

turn_id_mapping = {
key: index for index, key in enumerate(turn_maneuver)
}

turn_map = {
    # 0: "no_lane",
    0: "no_junction",
    1: "turn_left",
    2: "turn_right"
}

preceding_maneuver = {
    0:"no_preceding",
    1:"follow",
    2:"approach",
    # 3:"fall_behind"

}

preceding_id_mapping = {
key: index for index, key in enumerate(preceding_maneuver)
}

preceding_map = {
    0:"no_preceding",
    1:"follow",
    2:"approach",
    # 3:"fall_behind"
}

categorymap = {"preceding": 3,"turn":3,"lane":3,"vehiclestate":6}

