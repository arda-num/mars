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

def apply_left_turn(batch_obj_dyn, actor_id, angle_per_frame, x_offset_per_frame, y_offset_per_frame, z_offset_per_frame):
    rotation = batch_obj_dyn[..., 3]
    rotation[:, :, actor_id] += angle_per_frame
    batch_obj_dyn[..., 3] = rotation

    pose = batch_obj_dyn[..., :3]
    pose[:, :, actor_id, 0] += x_offset_per_frame
    pose[:, :, actor_id, 1] += y_offset_per_frame
    pose[:, :, actor_id, 2] += z_offset_per_frame
    batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn


def apply_right_turn(batch_obj_dyn, actor_id, angle_per_frame, x_offset_per_frame, y_offset_per_frame, z_offset_per_frame):
    rotation = batch_obj_dyn[..., 3]
    rotation[:, :, actor_id] -= angle_per_frame
    batch_obj_dyn[..., 3] = rotation

    pose = batch_obj_dyn[..., :3]
    pose[:, :, actor_id, 0] += x_offset_per_frame
    pose[:, :, actor_id, 1] += y_offset_per_frame
    pose[:, :, actor_id, 2] += z_offset_per_frame
    batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn

def apply_sudden_stop(batch_obj_dyn, actor_id, deceleration_per_frame):
    velocity = batch_obj_dyn[..., 5]  # Assuming the last dimension represents velocity
    velocity[:, :, actor_id] -= deceleration_per_frame
    batch_obj_dyn[..., 5] = velocity

    return batch_obj_dyn

