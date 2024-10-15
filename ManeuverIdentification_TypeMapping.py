from enum import Enum


class LaneletGeometry(Enum):
    straight = 'straight'
    left = 'left'
    right = 'right'


maneuver_type_list = ['vehicle_state', 'lane', 'turn', 'park', 'preceding', 'passing']


class ManeuverType(Enum):
    vehicle_state = 'vehicle_state'
    lane = 'lane'
    turn = 'turn'
    park = 'park'
    preceding = 'preceding'
    passing = 'passing'


class ClassificationState(Enum):
    pass


class VehicleStateManeuvers(ClassificationState):
    """
    Maneuvers regarding the driving state of the vehicle
    - accelerate: increasing speed
    - driveaway: acceleration from standstill
    - keep_velocity
    - standstill: keep_velocity at 0
    - decelerate: decrease speed
    - halt: decelrate until standstill
    - reversing
    """
    accelerate = 1
    driveaway = 11
    keep_velocity = 2
    standstill = 21
    decelerate = 3
    halt = 31
    reversing = 4  # todo: implement


class LaneManeuvers(ClassificationState):
    """
    Maneuver regarding Lange changes

    - no_lane: no lanelet can be mapped to point thus not on a road/lane
    - follow_lane: proceed to drive on lane or change to next lane segment
    - lane_change: changing lanes laterally
    """
    no_lane = -1
    follow_lane = 1
    lane_change = 2


class TurnManeuvers(ClassificationState):
    """
    Maneuvers based on the road shape

    - no_lane: no lanelet can be mapped to point thus not on a road/lane
    - no_junction: currently not on a junction lanelet (lanelet does not overlay with any other lanelet)
    - turn_left: on a junction lanelet and lanelet is curved to the left
    - turn_right: on a junction lanelet and lanelet is curved to the right
    - u_turn: on a junction lanelet which performs a u-turn
    """
    no_lane = -1
    no_junction = 0
    turn_left = 1
    turn_right = 2
    u_turn = 3  # todo: implement


class ParkManeuvers(ClassificationState):  # todo: implement
    """
    Whether or not a vehicle is currently parking

    - no_park
    - park
    """
    no_park = 0
    park = 1


class PrecedingManeuvers(ClassificationState):
    """
    Maneuvers based on a preceding vehicle (a vehicle driving directly in front
    of the ego-vehicle)

    - no_preceding: no vehicle in front
    - follow: relative speed about 0
    - approach: ego is approaching the vehicle in front
    - fall_behind: ego is falling behind the vehilce in front
    """
    no_preceding = 0
    follow = 1
    approach = 2
    fall_behind = 3


class PassingManeuvers(ClassificationState):  # todo: implement
    """
    Maneuvers regarding vehicles on an adjacent lane

    - no_passing: currently not passing a vehicle
    - passing: passing a vehicle on an adjacent lane
    """
    no_passing = 0
    passing = 1


infrastructure_context_types = ['junction', 'crosswalk']


class InfrastructureContextType(Enum):
    """
    Signals referencing a context object in the scenario
    """
    junction = 'junction'
    crosswalk = 'crosswalk'


class JunctionContextTypes(ClassificationState):  # todo: implement
    """
    Signal referencing the ego-vehicles state towards junctions

    - no_junction: no junction is currently in the future path of the vehicle
    - approach: approaching a junction
    - cross: crossing a junction
    """
    no_junction = 0
    approach = 1
    cross = 2


class CrosswalkContextTypes(ClassificationState):
    """
    Signal referencing the ego-vehicles state towards crosswalk

    - no_crosswalk: no crosswalk is currently in the future path of the vehicle
    - approach: approaching a crosswalk
    - cross: crossing a crosswalk
    """
    no_crosswalk = 0
    approach = 1
    cross = 0


class InteractingVehicleTypes(Enum):
    leading = 'leading'
    crossing = 'crossing'
    close = 'close'
