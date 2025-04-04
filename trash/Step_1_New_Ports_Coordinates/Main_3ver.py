import pyvista as pv
import numpy as np
import Tools.Function_for_DATA_angel as FuD
import coordinate_output_NBI_and_ports as Cout
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

def find_first_two_intersections(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    intersections = surface.ray_trace(point, ray_end)[0]
    return intersections[:2]  

def find_intersection(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    return surface.ray_trace(point, ray_end)[0][0]

def get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface):
    new_NBI_start, new_NBI_end, lines = [], [], []
    NBI_P1 = np.array([NBI_X, NBI_Y, NBI_Z])
    NBI_P2 = np.array([NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z])
  
    for i in range(NBI_P1.shape[1]):
        start_point, direction_vector = NBI_P1[:, i], NBI_P2[:, i]
        intersection1, intersection2 = find_first_two_intersections(surface, start_point, direction_vector)
        new_NBI_start.append(intersection1)
        new_NBI_end.append(intersection2)
        lines.append(pv.Line(intersection1, intersection2))

    return np.array(new_NBI_start).T, np.array(new_NBI_end).T, lines

def get_intersection_points(P_1, P_2, surface):
    new_P_1, lines = [], []
    for i in range(P_1.shape[1]):
        start_point = P_1[:, i]
        direction_vector = P_2[:, i] - start_point
        intersection = find_intersection(surface, start_point, direction_vector)
        new_P_1.append(intersection) 
        line = pv.Line(start_point, intersection)
        lines.append(line)
    new_P_1 = np.array(new_P_1).T
    return new_P_1, lines

def check_intersection(point, candidate_point, surface):
    intersection_points = surface.ray_trace(point, candidate_point)[0]
    if len(intersection_points) > 0 and np.linalg.norm(intersection_points[0] - point) < 1e-3:
        return intersection_points[1:]  
    return intersection_points

def find_max_valid_range(new_P_1, NBI_start, NBI_end, surface):
    mid_point = (NBI_start + NBI_end) / 2
    valid_indices, extreme_points_1, extreme_points_2, valid_lines = [], [], [], []

    def find_extreme_points(point, direction, mid_point, NBI_limit):
        max_valid_point = mid_point
        step_vector = direction / np.linalg.norm(direction)
        for t in np.linspace(0, 1, 100):
            candidate_point = mid_point + t * step_vector * np.linalg.norm(direction)
            if len(check_intersection(point, candidate_point, surface)) > 0:
                break
            max_valid_point = candidate_point
        return max_valid_point

    for i, point in enumerate(new_P_1.T):
        if len(check_intersection(point, mid_point, surface)) == 0:
            valid_indices.append(i)
            direction_to_start = NBI_start - mid_point
            direction_to_end = NBI_end - mid_point
            max_start = find_extreme_points(point, direction_to_start, mid_point, NBI_start)
            max_end = find_extreme_points(point, direction_to_end, mid_point, NBI_end)
            extreme_points_1.append(max_start)
            extreme_points_2.append(max_end)
            valid_lines.extend([pv.Line(point, mid_point), pv.Line(point, max_start), pv.Line(point, max_end)])

    return valid_indices, extreme_points_1, extreme_points_2, valid_lines

def add_valid_points_to_plotter(plotter, points, indices, color='blue', point_size=10):
    valid_points = points[:, indices].T
    plotter.add_mesh(pv.PolyData(valid_points), color=color, point_size=point_size, render_points_as_spheres=True)

def add_labels(plotter, points, labels, text_color='white', point_color='blue'):
    plotter.add_point_labels(pv.PolyData(points.T), labels, point_size=10, font_size=12, text_color=text_color, point_color=point_color)

def NBI_and_PORTS(NBI_index, lines2, new_P_1, new_NBI_start, new_NBI_end, surface, plotter, P_name):
    plotter.add_mesh(lines2[NBI_index], color='red', line_width=3)
    NBI_start, NBI_end = new_NBI_start[:, NBI_index], new_NBI_end[:, NBI_index]
    valid_indices, extreme_points_1, extreme_points_2, valid_lines = find_max_valid_range(new_P_1, NBI_start, NBI_end, surface)

    add_valid_points_to_plotter(plotter, new_P_1, valid_indices, color='red', point_size=12)
    NBI_labels = [f"NBI {i}" for i in range(new_NBI_start.shape[1])]
    add_labels(plotter, new_NBI_start, NBI_labels, text_color='white', point_color='red')
    # Uncomment to add port labels if needed
    # valid_port_names = [P_name[i] for i in valid_indices]
    # add_labels(plotter, new_P_1[:, valid_indices], valid_port_names)

    print(f"New points for ports were found")

if __name__ == "__main__":
    start_time = time.time()

    # Load data
    R_x, R_y, R_z = FuD.all_point(FuD.read_data()[0])
    P_1, P_2, P_name = Cout.Ports()
    NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z = Cout.NBI()

    # Create surface
    surface = create_surface(R_x, R_y, R_z)
    
    # Initialize plotter and add surface
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan', show_edges=True, opacity=0.2)
    
    # Get intersections for ports and NBI
    new_P_1, lines1 = get_intersection_points(P_1, P_2, surface)
    new_NBI_start, new_NBI_end, lines2 = get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface)

    # Add NBI and ports to plot
    NBI_index = 1
    NBI_and_PORTS(NBI_index, lines2, new_P_1, new_NBI_start, new_NBI_end, surface, plotter, P_name)
    
    plotter.show()

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
