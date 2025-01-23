import pyvista as pv
import numpy as np
import Surface_data as FuD
import NBI_Ports_data_input as Cout
import time

def create_surface(R_x, R_y, R_z):
    points, faces = [], []
    for i in range(len(R_x) - 1):
        current_contour = np.column_stack((R_x[i], R_y[i], R_z[i]))
        next_contour = np.column_stack((R_x[i+1], R_y[i+1], R_z[i+1]))
        for j in range(len(current_contour)):
            p1, p2 = current_contour[j], current_contour[(j+1) % len(current_contour)]
            n1, n2 = next_contour[j], next_contour[(j+1) % len(current_contour)]
            points.extend([p1, p2, n2, n1])
            faces.append([4, len(points)-4, len(points)-3, len(points)-2, len(points)-1])
    return pv.PolyData(np.array(points), faces=np.hstack(faces))

def find_intersection(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    return surface.ray_trace(point, ray_end)[0][0]

def find_first_two_intersections(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    intersections = surface.ray_trace(point, ray_end)[0]
    return intersections[0], intersections[1]  




def get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface):
    new_NBI_start, new_NBI_end, lines = [], [], []
    NBI_P1 = np.array([NBI_X, NBI_Y, NBI_Z])
    NBI_P2 = np.array([NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z])
  
    for i in range(NBI_P1.shape[1]):
        start_point, direction_vector = NBI_P1[:, i], NBI_P2[:, i]
        intersection1, intersection2 = find_first_two_intersections(surface, start_point, direction_vector)


        
        # Вектор от intersection1 к intersection2
        direction_vector = intersection2 - intersection1
        
        # Нормализуем вектор
        direction_normalized = direction_vector / np.linalg.norm(direction_vector)

        # Смещаем обе точки внутрь на заданное расстояние
        intersection1_shifted = intersection1 + direction_normalized * 20
        intersection2_shifted = intersection2 - direction_normalized * 20
        
        new_NBI_start.append(intersection1_shifted)
        new_NBI_end.append(intersection2_shifted)
        lines.append(pv.Line(intersection1_shifted, intersection2_shifted))

    return np.array(new_NBI_start).T, np.array(new_NBI_end).T, lines

def get_intersection_points(P_1,P_2, surface):
    new_P_1, lines = [], []
    for i in range(P_1.shape[1]):
        start_point = P_1[:, i]
        direction_vector = P_2[:, i] - start_point
        intersection = find_intersection(surface, start_point, direction_vector)
        new_P_1.append(intersection) 
        lines.append(pv.Line(start_point, intersection))
    return np.array(new_P_1).T, lines

def check_intersection(point, candidate_point, surface):
    intersection_points = surface.ray_trace(point, candidate_point)[0]
    if len(intersection_points) > 0 and np.linalg.norm(intersection_points[0] - point) < 1e-3:
        return intersection_points[1:]  
    else:
     return intersection_points  
    
def find_extreme_points(point_1, point, direction, mid_point, NBI_limit, surface, angle):
        max_valid_point = mid_point
        step_vector = direction / np.linalg.norm(direction)
        candidate_point = mid_point
        for t in np.linspace(0, 1, 100):
            candidate_point = mid_point + t * step_vector * np.linalg.norm(direction)
            if len(check_intersection(point, candidate_point, surface)) > 0 or check_segment_angle(point_1, point, candidate_point)>angle:
                break
            max_valid_point = candidate_point
            if t == 90:
                break
        return max_valid_point
import numpy as np

def check_segment_angle(point_1, point_2, point_checked):

    vector_AB = np.array(point_2) - np.array(point_1)  
    vector_CD = np.array(point_checked) - np.array(point_2)  
    
    dot_product = np.dot(vector_AB, vector_CD)
    

    magnitude_AB = np.linalg.norm(vector_AB)
    magnitude_CD = np.linalg.norm(vector_CD)
    

    if magnitude_AB == 0 or magnitude_CD == 0: 
        return None 

    cos_theta = dot_product / (magnitude_AB * magnitude_CD)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    
    angle_deg = np.degrees(angle_rad)
    #print(angle_deg)
    return angle_deg

def check_angle_2_vec(vector_AB, vector_CD):
    
    dot_product = np.dot(vector_AB, vector_CD)
    

    magnitude_AB = np.linalg.norm(vector_AB)
    magnitude_CD = np.linalg.norm(vector_CD)
    

    if magnitude_AB == 0 or magnitude_CD == 0: 
        return None 

    cos_theta = dot_product / (magnitude_AB * magnitude_CD)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_theta)
    
    angle_deg = np.degrees(angle_rad)
    #print(angle_deg)
    return angle_deg



def find_max_valid_range(P_1, new_P_1, NBI_start, NBI_end, surface, angle):
    mid_point = (NBI_start + NBI_end) / 2
    valid_indices, extreme_points_1, extreme_points_2, valid_lines = [], [], [], []
    if isinstance(new_P_1[1], np.ndarray):
     for i, point in enumerate(new_P_1.T):
        point_1 = [P_1[0][i],P_1[1][i], P_1[2][i]]
        if len(check_intersection(point, mid_point, surface)) == 0 and float(check_segment_angle(point_1, point, mid_point))<=angle:
            valid_indices.append(i)
            if i ==57:
                print(check_segment_angle(point_1, point, mid_point), point_1, point)
            direction_to_start = NBI_start - mid_point
            direction_to_end = NBI_end - mid_point
            max_start = find_extreme_points(point_1, point, direction_to_start, mid_point, NBI_start, surface, angle)
            max_end = find_extreme_points(point_1, point, direction_to_end, mid_point, NBI_end, surface, angle)
            extreme_points_1.append(max_start)
            extreme_points_2.append(max_end)
            valid_lines.extend([pv.Line(point, max_start), pv.Line(point, max_end)])
    else:
        point =  [new_P_1[0], new_P_1[1], new_P_1[2]]      
        point_1 = [P_1[0],P_1[1], P_1[2]]
        if len(check_intersection(point, mid_point, surface)) == 0 and float(check_segment_angle(point_1, point, mid_point))<=angle:
            valid_indices.append(0)
            direction_to_start = NBI_start - mid_point
            direction_to_end = NBI_end - mid_point
            max_start = find_extreme_points(point_1, point, direction_to_start, mid_point, NBI_start, surface, angle)
            max_end = find_extreme_points(point_1, point, direction_to_end, mid_point, NBI_end, surface, angle)
            extreme_points_1.append(max_start)
            extreme_points_2.append(max_end)
            valid_lines.extend([pv.Line(point, max_start), pv.Line(point, max_end)])

    return valid_indices, extreme_points_1, extreme_points_2, valid_lines

def pre_find_max_valid_range(new_P_1, NBI_start, NBI_end, surface):
    mid_point = (NBI_start + NBI_end) / 2
    valid_indices = []
    for i, point in enumerate(new_P_1.T):
        if len(check_intersection(point, mid_point, surface)) == 0:
            valid_indices.append(i)
    return valid_indices


def add_points_to_plotter(plotter, points, color='blue', point_size=10):
    points = np.array(points)
    point_cloud = pv.PolyData(points)
    plotter.add_mesh(point_cloud, color=color, point_size=point_size, render_points_as_spheres=True)

def add_valid_points_to_plotter(plotter, points, indices, color='blue', point_size=10):
    valid_points = points[:, indices].T
    plotter.add_mesh(pv.PolyData(valid_points), color=color, point_size=point_size, render_points_as_spheres=True)

def add_labels(plotter, points, labels, text_color='white', point_color='blue'):
    plotter.add_point_labels(pv.PolyData(points.T), labels, point_size=10, font_size=12, text_color=text_color, point_color=point_color)


def NBI_and_PORTS(P_1, NBI_index, new_P_1,new_NBI_start, new_NBI_end, surface, angle):
    NBI_start, NBI_end = new_NBI_start[:, NBI_index], new_NBI_end[:, NBI_index]
    valid_indices, extreme_points_1, extreme_points_2, valid_lines = find_max_valid_range(P_1, new_P_1, NBI_start, NBI_end, surface, angle)
    return valid_indices, extreme_points_1, extreme_points_2, valid_lines

def pre_NBI_and_PORTS(NBI_index, new_P_1,new_NBI_start, new_NBI_end, surface):
    NBI_start, NBI_end = new_NBI_start[:, NBI_index], new_NBI_end[:, NBI_index]
    valid_indices = pre_find_max_valid_range(new_P_1, NBI_start, NBI_end, surface)
    return valid_indices

def visualisation(surface,lines1, lines2,NBI_index,new_P_1,valid_indices, new_NBI_start,extreme_points_1,extreme_points_2, valid_lines, P_name):
    # Initialize plotter and add surface
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan', show_edges=True, opacity=0.2)
    plotter.add_mesh(lines2[NBI_index], color='red', line_width=3) 
    add_valid_points_to_plotter(plotter, new_P_1, valid_indices, color='red', point_size=12)
    NBI_labels = [f"NBI {i}" for i in range(new_NBI_start.shape[1])]  
    add_labels(plotter, new_NBI_start, NBI_labels)


    #add_points_to_plotter(plotter, new_P_1.T, color='yellow', point_size=12)
    #add_points_to_plotter(plotter, extreme_points_1, color='red', point_size=15)
    #add_points_to_plotter(plotter, extreme_points_2, color='blue', point_size=15)
    for line in valid_lines:
        plotter.add_mesh(line, color='green', line_width=1)
    # Get the port names for valid indices
    valid_port_names = [P_name[i] for i in valid_indices]
    add_labels(plotter, new_P_1[:, valid_indices], valid_port_names)
    #add_labels(plotter, new_P_1, P_name)


    #for line in lines1:
     #   plotter.add_mesh(line, color='yellow', line_width=3)

    for line in lines2:
        plotter.add_mesh(line, color='red', line_width=3)
    plotter.show()


if __name__ == "__main__":
    start_time = time.time()
    # Load data
    R_x, R_y, R_z = FuD.all_point(FuD.read_data()[0])
    P_1, P_2, P_name = Cout.Ports()
    NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z = Cout.NBI()
    
    # Create 3D surface
    surface_start_time = time.time()  
    surface = create_surface(R_x, R_y, R_z)
    surface_end_time = time.time()  
    print(f"Surface creation took {surface_end_time - surface_start_time:.2f} seconds")


    #Get intersections for ports and NBI
    new_P_2_time_start = time.time()  
    new_P_1, lines1 = get_intersection_points(P_1,P_2, surface)
    new_P_2_time_end = time.time()  
    print(f"Ports processing took {new_P_2_time_end - new_P_2_time_start:.2f} seconds")

    #get new start and end NBI
    NBI_start_time = time.time()  
    new_NBI_start, new_NBI_end, lines2 = get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface)
    NBI_end_time = time.time()  
    print(f"NBI processing took {NBI_end_time - NBI_start_time:.2f} seconds")

     # Add NBI and ports to plot
    find_good_ports_start_time = time.time() 
    NBI_index = 1
    angle = 90
    valid_indices, extreme_points_1, extreme_points_2, valid_lines = NBI_and_PORTS(P_1, NBI_index, new_P_1,new_NBI_start, new_NBI_end, surface, angle)
    find_good_ports_end_time = time.time()  
    print(valid_indices)
    print(f"Valid ports processing {find_good_ports_end_time - find_good_ports_start_time:.2f} seconds")
    


    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    visualisation(surface,lines1, lines2,NBI_index,new_P_1,valid_indices, new_NBI_start,extreme_points_1,extreme_points_2, valid_lines, P_name)

    

    









