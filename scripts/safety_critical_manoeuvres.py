def get_actor_index(batch_obj_dyn, actor_id):
    """
    Get the index of the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
    
    Returns:
        int: The index of the actor in the tensor, or -1 if not found.
    """
    object_ids = batch_obj_dyn[..., 4]
    indices = (object_ids == actor_id).nonzero(as_tuple=True)
    if len(indices[0]) > 0:
        return indices[2][0].item()  # Return the first matching index 
    return -1

def get_actor_coordinates(batch_obj_dyn, actor_index):
    """
    Get the x, y, z coordinates of the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.

    Returns:
        tuple: A tuple containing the (x, y, z) coordinates of the actor, or None if not found.
    """
    if actor_index == -1:
        print(f"Actor with index {actor_index} not found in the batch.")
        return None

    pose = batch_obj_dyn[..., :3]
    x_coordinate = pose[:, :, actor_index, 0]
    y_coordinate = pose[:, :, actor_index, 1]
    z_coordinate = pose[:, :, actor_index, 2]

    return (x_coordinate, y_coordinate, z_coordinate)


def apply_right_turn(
        batch_obj_dyn,
        actor_id,
        angle_per_frame,
        x_offset_per_frame,
        y_offset_per_frame,
        z_offset_per_frame,
        max_rotation,
        maneuver_frame,
    ):
    """
    Apply a right turn to the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
        angle (float): The angle to rotate.
        x_offset (float): The total offset to apply along the x-axis.
        y_offset (float): The total offset to apply along the y-axis.
        z_offset (float): The total offset to apply along the z-axis.
        max_rotation (float): The maximum rotation angle for the turn.
        maneuver_frame (int): The frame number of the current maneuver.
    """
    angle = angle_per_frame * maneuver_frame
    x_offset = x_offset_per_frame * maneuver_frame
    y_offset = y_offset_per_frame * maneuver_frame
    z_offset = z_offset_per_frame * maneuver_frame


    # Get current rotation of the actor
    rotation = batch_obj_dyn[..., 3]

    # Apply rotation and translation only if the current rotation is less than max_rotation
    if abs(angle) < abs(max_rotation):
        rotation[:, :, actor_id] -= angle

        pose = batch_obj_dyn[..., :3]
        pose[:, :, actor_id, 0] += x_offset
        pose[:, :, actor_id, 1] += y_offset
        pose[:, :, actor_id, 2] -= z_offset
        batch_obj_dyn[..., :3] = pose
    else:
        rotation[:, :, actor_id] -= max_rotation
        batch_obj_dyn[..., 3] = rotation
        # Stop x translation if maximum rotation is reached
        pose = batch_obj_dyn[..., :3]
        pose[:, :, actor_id, 1] += y_offset
        pose[:, :, actor_id, 2] -= z_offset
        batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn
    
def apply_left_lane_shift(
        batch_obj_dyn,
        actor_id,
        angle_per_frame,
        z_offset_per_frame,
        max_rotation,
        maneuver_frame,
        total_frames,
        **kwargs,
    ):
    """
    Apply a left lane shift to the actor with the given actor_id.
    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
        angle_per_frame (float): The angle to rotate per frame.
        z_offset_per_frame (float): The offset to apply along the z-axis per frame.
        max_rotation (float): The maximum rotation angle for the lane shift.
        maneuver_frame (int): The frame number of the current maneuver.
        total_frames (int): The total number of frames for the maneuver.
    """
    if maneuver_frame < int(total_frames / 2): 
        angle = angle_per_frame * maneuver_frame
    elif int(total_frames / 2) <= maneuver_frame and maneuver_frame <= total_frames:
        angle = angle_per_frame * (total_frames - maneuver_frame)
    else:
        angle = 0.0
    
    if abs(angle) > abs(max_rotation):
        angle = abs(max_rotation) if angle > 0 else -abs(max_rotation)

    z_offset = None
    if z_offset_per_frame < 0:
        z_offset = max(z_offset_per_frame * maneuver_frame, -0.5)
    else:
        z_offset = min(z_offset_per_frame * maneuver_frame, 0.5)

    # Get current rotation of the actor
    rotation = batch_obj_dyn[..., 3]

    # Apply rotation
    rotation[:, :, actor_id] += angle
    pose = batch_obj_dyn[..., :3]
    pose[:, :, actor_id, 2] += z_offset
    batch_obj_dyn[..., :3] = pose


    return batch_obj_dyn

def apply_sudden_stop(
        batch_obj_dyn,
        actor_id,
        total_frames,
        maneuver_frame,
        initial_positions=None,
        **kwargs,
    ):
    """
    Apply a sudden stop to the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
        total_frames (int): The total number of frames for the maneuver.
        maneuver_frame (int): The frame number of the current maneuver.
        initial_positions (tuple, optional): The initial (x, z) positions where the stop maneuver starts.
                                             If None, it will be fetched from batch_obj_dyn during the first frame.
    """

    stopping_distance = 0.5  # The distance to stop the actor
    if initial_positions is None:
        initial_positions = get_actor_coordinates(batch_obj_dyn, actor_id)
        if initial_positions is None:
            return batch_obj_dyn, None  # Actor not found

    initial_x_position, _, initial_z_position = initial_positions

    # Calculate the deceleration factor based on the maneuver frame
    deceleration_factor = max(0,(total_frames - maneuver_frame) / total_frames)
    print("deceleration_factor: ", deceleration_factor)
    # Calculate the new positions
    current_x_position = initial_x_position + stopping_distance * (1 - deceleration_factor)
    current_z_position = initial_z_position
    print("distance: ", stopping_distance * (1 - deceleration_factor))
    pose = batch_obj_dyn[..., :3]
    pose[:, :, actor_id, 0] = current_x_position
    pose[:, :, actor_id, 2] = current_z_position
    batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn, initial_positions