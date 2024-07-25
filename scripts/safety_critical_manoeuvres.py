import torch 

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

def get_actor_coordinates(batch_obj_dyn, actor_id):
    """
    Get the x, y, z coordinates of the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.

    Returns:
        tuple: A tuple containing the (x, y, z) coordinates of the actor, or None if not found.
    """
    
    actor_index = get_actor_index(batch_obj_dyn, actor_id)
    if actor_index == -1:
        print(f"Actor with index {actor_index} not found in the batch.")
        return None

    pose = batch_obj_dyn[..., :3]
    x_coordinate = pose[:, :, actor_index, 0]
    y_coordinate = pose[:, :, actor_index, 1]
    z_coordinate = pose[:, :, actor_index, 2]

    return (x_coordinate, y_coordinate, z_coordinate)

def _pick_the_closest_car_front_along_x_axis(batch_obj_dyn, reference_actor_id, z_tolerance=0.3):
    # Get the coordinates of the reference actor
    reference_coordinates = get_actor_coordinates(batch_obj_dyn, reference_actor_id)
    reference_x_position, _, reference_z_position = reference_coordinates

    # List to store IDs of cars behind the reference actor
    closest_car_id = None
    minimum_distance = float('inf')
    
    actor_ids = torch.flatten(batch_obj_dyn[..., 4])
    # Iterate over all actors and check if any are behind the reference actor along the x-axis and within z_tolerance
    for i, actor_id in enumerate(actor_ids): 
        if actor_id == 0: continue #empty actor_id
        if actor_id != reference_actor_id:  # Skip the reference actor itself
            actor_coordinates = get_actor_coordinates(batch_obj_dyn, actor_id)
            actor_x_position, _, actor_z_position = actor_coordinates
            actor_id = batch_obj_dyn[0, 0, i, 4]
            if actor_x_position > reference_x_position and abs(actor_z_position - reference_z_position) <= z_tolerance:
                distance = (actor_x_position - reference_x_position) 
                print(f"Actor with ID {actor_id} is front of the reference actor with ID {reference_actor_id} along the x-axis with distance {distance}.")
                if distance < minimum_distance: 
                    print(f"MINIMUM!")
                    minimum_distance = distance
                    closest_car_id = actor_id
    return closest_car_id, float(minimum_distance)

def check_car_behind_along_x_axis(batch_obj_dyn, new_batch_obj_dyn, reference_actor_id, z_tolerance=0.3):
    """
    Check if there exists any car behind the reference car along the x-axis.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        reference_actor_id (int): The actor ID of the reference actor.
        z_tolerance (float): The tolerance width along the z-axis to consider cars being behind.

    Returns:
        int: The ID of the car behind the reference car along the x-axis, or -1 if none found.
    """

    # Get the coordinates of the reference actor
    try:
        reference_coordinates = get_actor_coordinates(batch_obj_dyn, reference_actor_id)
    except Exception as ex: 
        return []
    reference_x_position, _, reference_z_position = reference_coordinates

    # List to store IDs of cars behind the reference actor
    cars_behind = []
    
    actor_ids = torch.flatten(new_batch_obj_dyn[..., 4])
    print("actor_ids: ", actor_ids)
    # Iterate over all actors and check if any are behind the reference actor along the x-axis and within z_tolerance
    for i, actor_id in enumerate(actor_ids): 
        if actor_id == 0: continue #empty actor_id
        if actor_id != reference_actor_id:  # Skip the reference actor itself
            actor_coordinates = None
            actor_coordinates = get_actor_coordinates(batch_obj_dyn, actor_id)
            if actor_coordinates is None:
                actor_coordinates = get_actor_coordinates(new_batch_obj_dyn, actor_id)
            actor_x_position, _, actor_z_position = actor_coordinates
            actor_id = batch_obj_dyn[0, 0, i, 4]
            if actor_x_position < reference_x_position and abs(actor_z_position - reference_z_position) <= z_tolerance:
                print(f"Actor with ID {actor_id} is behind the reference actor with ID {reference_actor_id} along the x-axis.")
                cars_behind.append(int(actor_id))
    print("********")
    return cars_behind
    
def apply_turn(
        batch_obj_dyn,
        actor_id,
        angle_per_frame,
        x_offset_per_frame,
        z_offset_per_frame, 
        max_rotation,
        maneuver_frame,
        total_frames,
        **kwargs,
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
            maneuver_frame (int): The frame number of the current maneuver.
        """
        angle = angle_per_frame * maneuver_frame
        if abs(angle) > abs(max_rotation):
            angle = abs(max_rotation) if angle > 0 else -abs(max_rotation)
            
        x_offset = x_offset_per_frame * maneuver_frame
        z_offset = z_offset_per_frame * maneuver_frame
        
        # Get current rotation of the actor
        rotation = batch_obj_dyn[..., 3]
    
        # Apply rotation and translation only if the current rotation is less than max_rotation
        actor_index = get_actor_index(batch_obj_dyn, actor_id)
        rotation[:, :, actor_index] += angle
        pose = batch_obj_dyn[..., :3]
        pose[:, :, actor_index, 0] += x_offset
        pose[:, :, actor_index, 2] += z_offset
        batch_obj_dyn[..., :3] = pose
        
        return batch_obj_dyn 

def apply_lane_shift(
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

    actor_index = get_actor_index(batch_obj_dyn, actor_id)
    # Apply rotation
    rotation[:, :, actor_index] += angle
    pose = batch_obj_dyn[..., :3]
    pose[:, :, actor_index, 2] += z_offset
    batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn
    
def save_batch_obj_dyn(modified_batch_obj_dyn, batch_obj_dyn, actor_id):
    """
    Save the modified positions for the specified actor in the modified_batch_obj_dyn tensor.

    Args:
        modified_batch_obj_dyn (torch.Tensor): The tensor storing modified positions for actors.
        batch_obj_dyn (torch.Tensor): The current batch object dynamics tensor.
        actor_id (int): The actual ID of the actor whose position needs to be saved.

    Returns:
        torch.Tensor: The updated modified_batch_obj_dyn tensor.
    """
    actor_index = get_actor_index(batch_obj_dyn, actor_id)
    if actor_index == -1:
        print(f"Actor with ID {actor_id} not found in the batch.")
        return modified_batch_obj_dyn

    # Ensure modified_batch_obj_dyn has the same shape as batch_obj_dyn
    if modified_batch_obj_dyn is None:
        modified_batch_obj_dyn = batch_obj_dyn

    # Save the actor's parameters
    modified_batch_obj_dyn[:, :, actor_index, :5] = batch_obj_dyn[:, :, actor_index, :5]

    return modified_batch_obj_dyn
    

def apply_sudden_stop(
        batch_obj_dyn,
        actor_id,
        total_frames,
        maneuver_frame,
        initial_positions=None,
        modified_batch_obj_dyn=None,
        **kwargs,
    ):
    """
    Apply a sudden stop to the actor with the given actor_id.

    Args:
        batch_obj_dyn (torch.Tensor): The batch object dynamics tensor.
        actor_id (int): The actual ID of the actor.
        total_frames (int): The total number of frames for the maneuver.
        maneuver_frame (int): The frame number of the current maneuver.
        initial_positions (dict, optional): The initial (x, z) positions where the stop maneuver starts for each actor_id.
                                             If empty, it will be fetched from batch_obj_dyn during the first frame.
    """
    
    stopping_distance = 0.8  # The distance to stop the actor
    if modified_batch_obj_dyn != None: # If modified rays info is provided 
        try:
            closest_car_id, minimum_distance = _pick_the_closest_car_front_along_x_axis(modified_batch_obj_dyn, actor_id) # find the closest car front
            if minimum_distance < stopping_distance:
                stopping_distance = minimum_distance - 0.1
        except Exception as ex:
            print("HALOOO2: ", ex)
    
    try: 
        initial_position = initial_positions[actor_id]
    except:
        initial_position = get_actor_coordinates(batch_obj_dyn, actor_id)
        initial_positions[actor_id] = initial_position
        if initial_position is None: # Should not be hitting here at all!
            return batch_obj_dyn, None  # Actor not found

    initial_x_position, initial_y_position, initial_z_position = initial_position
    # Calculate the deceleration factor based on the maneuver frame
    deceleration_factor = max(0,(total_frames - maneuver_frame) / total_frames)
    print("deceleration_factor: ", deceleration_factor)
    # Calculate the new positions
    current_x_position = initial_x_position + stopping_distance * (1 - deceleration_factor)
    current_z_position = initial_z_position
    print("initial position: ", initial_position)
    print("distance: ", stopping_distance * (1 - deceleration_factor))
    pose = batch_obj_dyn[..., :3]
    actor_index = get_actor_index(batch_obj_dyn, actor_id)
    pose[:, :, actor_index, 0] = current_x_position
    pose[:, :, actor_index, 1] = initial_y_position 
    pose[:, :, actor_index, 2] = current_z_position
    batch_obj_dyn[..., :3] = pose

    return batch_obj_dyn, initial_positions