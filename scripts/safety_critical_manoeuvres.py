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

def apply_left_turn(
        batch_obj_dyn,
        actor_id,
        angle,
        x_offset,
        y_offset,
        z_offset,
        max_rotation,
    ):
    """
    Apply a left turn to the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
        angle (float): The angle to rotate.
        x_offset (float): The total offset to apply along the x-axis.
        y_offset (float): The total offset to apply along the y-axis.
        z_offset (float): The total offset to apply along the z-axis.
        max_rotation (float): The maximum rotation angle for the turn.
        frames (int): The number of frames over which to apply the turn.
    """

    # Get current rotation of the actor
    rotation = batch_obj_dyn[..., 3]

    # Apply rotation and translation only if the current rotation is less than max_rotation
    if abs(angle) < abs(max_rotation):
        rotation[:, :, actor_id] += angle

        pose = batch_obj_dyn[..., :3]
        pose[:, :, actor_id, 0] += x_offset
        pose[:, :, actor_id, 1] += y_offset
        pose[:, :, actor_id, 2] += z_offset
        batch_obj_dyn[..., :3] = pose
    else:
        rotation[:, :, actor_id] += max_rotation
        batch_obj_dyn[..., 3] = rotation
        # Stop x translation if maximum rotation is reached
        pose = batch_obj_dyn[..., :3]
        pose[:, :, actor_id, 1] += y_offset
        pose[:, :, actor_id, 2] += z_offset
        batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn


def apply_right_turn(
        batch_obj_dyn,
        actor_id,
        angle,
        x_offset,
        y_offset,
        z_offset,
        max_rotation
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
    """

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

def apply_sudden_stop(batch_obj_dyn, actor_id, deceleration_per_frame):
    velocity = batch_obj_dyn[..., 5]  # Assuming the last dimension represents velocity
    velocity[:, :, actor_id] -= deceleration_per_frame
    batch_obj_dyn[..., 5] = velocity

    return batch_obj_dyn

