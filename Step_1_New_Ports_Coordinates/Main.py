import pyvista as pv
import numpy as np
import Tools.Function_for_DATA_angel as FuD
import coordinate_output_NBI_and_ports as Cout

def create_and_save_torus_surface(R_x, R_y, R_z, output_filename='surface.obj'):
    all_contours = []
    for section_index in range(len(R_x)):
        section_points = np.column_stack((R_x[section_index], R_y[section_index], R_z[section_index]))
        n_points = len(section_points)
        lines = np.array([n_points] + list(range(n_points)) + [0], dtype=np.int32)
        contour_lines = pv.PolyData(section_points, lines=lines)
        all_contours.append(contour_lines)
    points = []
    faces = []

    for i in range(len(all_contours) - 1):
        current_contour = all_contours[i].points
        next_contour = all_contours[i + 1].points

        n_points = len(current_contour)

        for j in range(n_points):
            current_p1 = current_contour[j]
            current_p2 = current_contour[(j + 1) % n_points]
            next_p1 = next_contour[j]
            next_p2 = next_contour[(j + 1) % n_points]
            points.extend([current_p1, current_p2, next_p2, next_p1])
            idx = len(points) - 4
            faces.append([4, idx, idx + 1, idx + 2, idx + 3])
    points = np.array(points)
    faces = np.hstack(faces)
    surface = pv.PolyData(points, faces)

    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan', show_edges=True)
    plotter.add_title('Toroidal Surface')

    surface.save(output_filename)
    print(f'Torus surface saved to {output_filename}')
    return surface, plotter  

def find_intersection(surface, point, direction):

    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize the direction

    # Create a ray extending in the direction from the point
    ray = pv.Line(point, point + direction * 1000) 

    # Perform ray tracing
    intersection_points, _ = surface.ray_trace(point, point + direction * 1000)

    # Return the first intersection point (guaranteed that all points have intersections)
    first_intersection = intersection_points[0]
    return first_intersection

if __name__ == "__main__":
    # Data for surface
    Phi, R_phi, Z_phi = FuD.read_data()
    R_x_all, R_y_all, Z_all = FuD.all_point(Phi)
    R_x, R_y, R_z = R_x_all, R_y_all, Z_all
    
    # Data for ports
    NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z = Cout.NBI()
    P_1, P_2, P_3, P_unit_vector = Cout.Ports()   
    
    # Create 3D surface
    surface, plotter = create_and_save_torus_surface(R_x, R_y, R_z)

    # Find intersections for all start points
    intersection_points_array = []
    lines = []

    for i in range(len(P_1[0])):  # Assuming P_1, P_2 are lists of points
        start_point = np.array([P_1[0][i], P_1[1][i], P_1[2][i]])  
        direction_vector = np.array([P_2[0][i], P_2[1][i], P_2[2][i]]) - start_point 
        
        # Find the intersection point
        intersection = find_intersection(surface, start_point, direction_vector)

        # Append intersection point to array
        intersection_points_array.append(intersection)
            
        # Create a line connecting the start point and the intersection point
        line = pv.Line(start_point, intersection)
        lines.append(line)

    # Convert the intersection points array into a format similar to P_1 (three lists for x, y, z)
    intersection_points_array = np.array(intersection_points_array)
    intersection_P_1 = [intersection_points_array[:, 0], intersection_points_array[:, 1], intersection_points_array[:, 2]]

    # Add all the lines to the plotter
    #for line in lines:
    #    plotter.add_mesh(line, color='yellow', line_width=3)

    # Show the plot with surface and lines
    plotter.show()
