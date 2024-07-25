configs = {
    "scenes" : {
        "0006": {
            2: { # actor_id
                "left_turn" : { # manoeuvres
                    "angle": 7.0,
                    "x_offset": -1.5,
                    "y_offset": 0.0,
                    "z_offset": 3.0,
                    "max_rotation": -1.0,
                    "frames_per_maneuver": 25,
                    "maneuver_starting_frame": 0,
                    "maneuver_ending_frame": None, 
                },
                'left_lane_shift': {
                    "angle": 0.7,
                    "x_offset": 0.0,
                    "y_offset": 0.0,
                    "z_offset": 0.6,
                    "max_rotation": 0.35,
                    "frames_per_maneuver": 10,
                    "maneuver_starting_frame": 0,
                    "maneuver_ending_frame": None,
                }, 
                'right_lane_shift': {
                    "angle": -0.5,
                    "x_offset": 0.0,
                    "y_offset": 0.0,
                    "z_offset": -0.3,
                    "max_rotation": -0.25,
                    "frames_per_maneuver": 10,
                    "maneuver_starting_frame": 0,
                    "maneuver_ending_frame": None,
                }, 
                'sudden_stop': {
                    "angle": 0.0,
                    "x_offset": 0.0,
                    "y_offset": 0.0,
                    "z_offset": 0.0,
                    "max_rotation": 0.0,
                    "frames_per_maneuver": 5,
                    "maneuver_starting_frame": 0,
                    "maneuver_ending_frame": None,
                },
            },
        }
    }
}
